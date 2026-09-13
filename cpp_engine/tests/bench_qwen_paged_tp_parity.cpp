// TP paged-versus-contiguous parity and throughput on a real model.
//
// Drives the Qwen3.5 engine through the batch API under tensor parallelism and
// runs exactly ONE KV mode per process. The launcher spawns a contiguous run
// and a paged run (same checkpoint, same prompts, same seed) and diffs the
// emitted token checksums: paged KV must produce byte-identical output to the
// contiguous arena, and the decode throughput must not regress.
//
// The batch API is used throughout because it is the one TP-safe path:
// batch_prefill() and batch_decode_step() auto-announce the collective the
// workers join, so rank 0 needs no manual worker_command_* calls (world size 1
// makes those a no-op anyway, so the single-GPU admission runs work unchanged).
//
// Running one mode per process also sidesteps two hazards of alternating modes
// inside a single TP group: the workspace is a single buffer per engine and is
// reconstructed per construction anyway, and a worker loop exits at Shutdown
// and cannot be re-entered.

#include "qwen_engine.hpp"
#include "cuda_ops.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string checkpoint;
    int layers = 0;          // 0 = all layers
    int max_context = 1024;  // per-sequence ceiling; bounds the pool/arena
    int tp_world = 1;
    int tp_rank = 0;
    int device = -1;         // -1 = same as tp_rank
    std::string nccl_id_path;
    int batch_size = 4;
    int decode_steps = 16;
    int prefill_chunk_tokens = 8192;
    bool prefix_cache = false;
    uint64_t prefix_cache_mb = 0;
    int shared_prefix_tokens = 0;
    // "contiguous" or "paged". One process runs exactly one.
    std::string mode = "contiguous";
    int kv_block_size = 16;
    // 0 = derive the paged budget from what the contiguous arena would reserve.
    uint64_t kv_cache_mb = 0;
    unsigned seed = 42;
};

void usage(const char* prog) {
    std::printf(
        "usage: %s <checkpoint_dir> [options]\n"
        "  --layers N        layer count (0 = all, default 0)\n"
        "  --max-context N   per-sequence ceiling (default 1024)\n"
        "  --tp-world N      TP world size (default 1)\n"
        "  --tp-rank N       TP rank (default 0)\n"
        "  --device N        CUDA device ordinal (default: same as tp-rank)\n"
        "  --nccl-id PATH    NCCL ID file path\n"
        "  --batch-size N    requests per run (default 4)\n"
        "  --decode N        decode steps per request (default 16)\n"
        "  --prefill-chunk N maximum prefill chunk (default 8192)\n"
        "  --prefix-cache    enable cross-request cache (paged mode)\n"
        "  --prefix-cache-mb N prefix-cache budget in MiB (default derived)\n"
        "  --shared-prefix N common leading prompt tokens (default 0)\n"
        "  --mode MODE       contiguous | paged (default contiguous)\n"
        "  --kv-block-size N paged block size (default 16)\n"
        "  --kv-cache-mb N   paged pool budget MiB (0 = match contiguous arena)\n"
        "  --seed N          prompt generation seed (default 42)\n",
        prog);
}

Options parse_args(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        std::exit(2);
    }
    Options opts;
    opts.checkpoint = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_int = [&](int& out) {
            if (i + 1 < argc) out = std::atoi(argv[++i]);
        };
        auto next_u64 = [&](uint64_t& out) {
            if (i + 1 < argc) out = std::strtoull(argv[++i], nullptr, 10);
        };
        auto next_string = [&](std::string& out) {
            if (i + 1 < argc) out = argv[++i];
        };
        if (arg == "--layers") next_int(opts.layers);
        else if (arg == "--max-context") next_int(opts.max_context);
        else if (arg == "--tp-world") next_int(opts.tp_world);
        else if (arg == "--tp-rank") next_int(opts.tp_rank);
        else if (arg == "--device") next_int(opts.device);
        else if (arg == "--batch-size") next_int(opts.batch_size);
        else if (arg == "--decode") next_int(opts.decode_steps);
        else if (arg == "--prefill-chunk") next_int(opts.prefill_chunk_tokens);
        else if (arg == "--prefix-cache") opts.prefix_cache = true;
        else if (arg == "--prefix-cache-mb") next_u64(opts.prefix_cache_mb);
        else if (arg == "--shared-prefix") next_int(opts.shared_prefix_tokens);
        else if (arg == "--kv-block-size") next_int(opts.kv_block_size);
        else if (arg == "--kv-cache-mb") next_u64(opts.kv_cache_mb);
        else if (arg == "--seed") next_int(reinterpret_cast<int&>(opts.seed));
        else if (arg == "--nccl-id") next_string(opts.nccl_id_path);
        else if (arg == "--mode") next_string(opts.mode);
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            std::exit(2);
        }
    }
    return opts;
}

// A deterministic prompt. Identical across ranks and across runs so the
// contiguous and paged arms see exactly the same tokens.
std::vector<int> make_prompt(int length, unsigned seed, int request_index,
                             int shared_prefix_tokens) {
    shared_prefix_tokens = std::clamp(shared_prefix_tokens, 0, length);
    std::vector<int> prompt =
        shared_prefix_tokens == 0
            ? std::vector<int>()
            : make_prompt(shared_prefix_tokens, seed, 0, 0);
    prompt.reserve(static_cast<size_t>(length));
    std::mt19937 rng(seed + static_cast<unsigned>(request_index + 1));
    std::uniform_int_distribution<int> dist(1, 1000);
    for (int i = shared_prefix_tokens; i < length; ++i) {
        prompt.push_back(dist(rng));
    }
    return prompt;
}

// A cheap but order-sensitive digest of the token stream. Two runs that agree
// on this are, to all practical purposes, token-exact.
uint64_t digest_tokens(const std::vector<std::vector<int>>& per_request) {
    uint64_t h = 1469598103934665603ULL;  // FNV offset basis
    for (const auto& tokens : per_request) {
        for (int t : tokens) {
            h ^= static_cast<uint64_t>(static_cast<uint32_t>(t));
            h *= 1099511628211ULL;
        }
        // Separate requests so reordering a whole request would still differ.
        h ^= 0xFF00ULL;
        h *= 1099511628211ULL;
    }
    return h;
}

// The contiguous KV arena cost for this batch/context combination, in bytes.
// We need it to size the paged pool to the same budget; building a probe
// engine reads the model once, which is exactly what the real run does anyway.
uint64_t derive_arena_bytes(const Options& opts) {
    pocket::QwenEngineOptions probe;
    probe.device = opts.device >= 0 ? opts.device : opts.tp_rank;
    probe.max_batch_size = opts.batch_size;
    probe.kv_cache_dtype = pocket::QwenKvCacheDType::Fp16;
    probe.prefix_cache = false;
    probe.kv_paged = false;
    if (opts.tp_world > 1) {
        probe.tp_world = opts.tp_world;
        probe.tp_rank = opts.tp_rank;
        probe.nccl_id_path = opts.nccl_id_path;
    }
    pocket::QwenEngine probe_engine(opts.checkpoint, probe, opts.layers,
                                  opts.max_context);
    return probe_engine.kv_cache_bytes();
}

}  // namespace

int main(int argc, char** argv) {
    if (!pocket::cuda_runtime_available()) {
        std::fprintf(stderr, "CUDA runtime not available\n");
        return 1;
    }

    try {
        const Options opts = parse_args(argc, argv);
        if (opts.mode != "contiguous" && opts.mode != "paged") {
            std::fprintf(stderr, "unknown --mode %s\n", opts.mode.c_str());
            return 2;
        }
        if (opts.prefill_chunk_tokens <= 0 || opts.shared_prefix_tokens < 0 ||
            opts.shared_prefix_tokens > std::min(256, opts.max_context - 1)) {
            std::fprintf(stderr, "invalid prefill or shared-prefix setting\n");
            return 2;
        }
        if (opts.prefix_cache && opts.mode != "paged") {
            std::fprintf(stderr, "--prefix-cache requires --mode paged\n");
            return 2;
        }

        pocket::QwenEngineOptions engine_opts;
        engine_opts.device = opts.device >= 0 ? opts.device : opts.tp_rank;
        engine_opts.max_batch_size = opts.batch_size;
        engine_opts.kv_cache_dtype = pocket::QwenKvCacheDType::Fp16;
        engine_opts.prefix_cache = opts.prefix_cache;
        engine_opts.prefill_chunk_tokens = opts.prefill_chunk_tokens;
        if (opts.tp_world > 1) {
            engine_opts.tp_world = opts.tp_world;
            engine_opts.tp_rank = opts.tp_rank;
            engine_opts.nccl_id_path = opts.nccl_id_path;
        }
        if (opts.mode == "paged") {
            // Budget the pool to the contiguous arena so the arms are
            // byte-for-byte comparable: the paged arm gets a block pool of the
            // same total bytes the arena reserves, just handed out per block.
            const uint64_t arena = derive_arena_bytes(opts);
            const uint64_t budget = opts.kv_cache_mb != 0
                ? opts.kv_cache_mb << 20
                : arena;
            engine_opts.kv_paged = true;
            engine_opts.kv_block_size = opts.kv_block_size;
            engine_opts.kv_cache_bytes = budget;
            if (opts.prefix_cache_mb != 0) {
                engine_opts.prefix_cache_bytes = opts.prefix_cache_mb << 20;
            }
        }

        pocket::QwenEngine engine(opts.checkpoint, engine_opts, opts.layers,
                                opts.max_context);
        engine.allocate_batch_slots(opts.batch_size);
        engine.warmup_tp();

        if (opts.tp_rank != 0) {
            // Workers exist to join collectives. Every command rank 0 issues is
            // something they execute; Shutdown is the last one, and it returns
            // here. No assertion is needed: the slot-free regression covers the
            // worker-side pool lifecycle, this bench only exercises throughput.
            engine.run_worker_loop();
            return 0;
        }

        // Deterministic prompts. `--shared-prefix` makes the leading tokens
        // identical across rows so the global cache can be measured directly.
        std::vector<std::unique_ptr<pocket::BatchedRequest>> owned;
        std::vector<pocket::BatchedRequest*> batch;
        std::vector<std::vector<int>> generated;
        const int prompt_len = std::min(256, opts.max_context - 1);
        const int decode = opts.decode_steps;
        for (int i = 0; i < opts.batch_size; ++i) {
            auto req = std::make_unique<pocket::BatchedRequest>();
            req->request_id = static_cast<uint64_t>(1000 + i);
            req->slot_id = engine.allocate_slot(req->request_id);
            if (req->slot_id < 0) {
                throw std::runtime_error("slot allocation failed");
            }
            req->prompt_tokens = make_prompt(
                prompt_len, opts.seed, i, opts.shared_prefix_tokens);
            req->sampling.max_new_tokens = decode;  // never the stopping bound
            req->sampling.ignore_eos = true;
            batch.push_back(req.get());
            owned.push_back(std::move(req));
            generated.emplace_back();
        }

        const pocket::BatchPrefillResult prefill = engine.batch_prefill(batch, 0);
        const pocket::QwenPrefixCacheStats prefix = engine.prefix_cache_stats();

        // Prefill's predicted token is part of the output stream (it lands in
        // req->last_token, the seed of the first decode step), so it must be in
        // the digest too or a prefill-only divergence would pass unnoticed.
        for (size_t i = 0; i < batch.size(); ++i) {
            generated[i].push_back(batch[i]->last_token);
        }

        // Measure decode: aggregate wall-time, then per-request token streams.
        double decode_seconds = 0.0;
        for (int step = 0; step < decode; ++step) {
            const pocket::BatchDecodeResult dec = engine.batch_decode_step(batch);
            decode_seconds += dec.seconds;
            for (size_t i = 0; i < batch.size(); ++i) {
                generated[i].push_back(dec.next_tokens[i]);
            }
        }

        const uint64_t digest = digest_tokens(generated);
        const double decode_s_per_step = decode_seconds / std::max(1, decode);
        const double req_per_sec =
            static_cast<double>(opts.batch_size) / decode_seconds;

        std::printf(
            "\nTP paged-vs-contiguous parity  mode=%s  batch=%d  tp=%d  "
            "context=%d  layers=%d\n"
            "  prompt_tokens=%d  prefill_tokens=%d  prefill_ms=%.3f  "
            "decode_steps=%d  decode_ms/step=%.3f  req/s=%.2f\n"
            "  prefix_shared=%d  prefix_global_hits=%d  prefix_global_misses=%d  "
            "prefix_cache_blocks=%d  prefix_cache_bytes=%llu/%llu  "
            "prefix_evictions=%d\n"
            "  checksum=%016llx\n",
            opts.mode.c_str(), opts.batch_size, opts.tp_world, opts.max_context,
            opts.layers == 0 ? 64 : opts.layers, prompt_len, prefill.total_tokens,
            prefill.seconds * 1000.0, decode, decode_s_per_step * 1000.0,
            req_per_sec, opts.shared_prefix_tokens, prefix.global_hits,
            prefix.global_misses, prefix.global_cached_blocks,
            static_cast<unsigned long long>(prefix.global_cache_bytes),
            static_cast<unsigned long long>(prefix.global_cache_budget_bytes),
            prefix.global_evictions, static_cast<unsigned long long>(digest));
        std::fflush(stdout);

        engine.worker_command_shutdown();
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return 1;
    }
}
