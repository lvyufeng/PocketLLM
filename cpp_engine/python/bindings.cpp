// Optional pybind11 bridge for the native PocketLLM engines.
//
// The module exposes value types and token-oriented engine methods only.  It does
// not expose vendor handles or tensors, so Python callers cannot depend on the
// CUDA/Ascend implementation details that remain inside pocket_cpp_core.

#include "persistent_engine.hpp"
#include "qwen_config.hpp"
#include "qwen_engine.hpp"
#include "qwen_weights.hpp"
#include "batch_scheduler.hpp"
#include "inference_engine.hpp"
#include "device_runtime.hpp"
#include "model_registry.hpp"
#include "qwen_layer_components.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <utility>

namespace py = pybind11;
using namespace pocket;
using namespace pocket::qwen_components;

namespace {

// Owns a Python callable that is handed to the scheduler, which stores it on a
// request and releases it from its own background thread once the request
// retires. A bare py::function would decref there without the GIL, so the
// callable is held behind a shared_ptr whose deleter reacquires it. Copying the
// holder only bumps the shared_ptr count, so the scheduler is free to copy the
// enclosing std::function without holding a GIL of its own.
class GilSafeCallable {
public:
    explicit GilSafeCallable(py::function fn)
        : fn_(new py::function(std::move(fn)), GilDeleter{}) {}

    // Runs on the scheduler thread with no GIL held. Exceptions are reported
    // and swallowed: letting one escape would cross the scheduler's thread and
    // take down every other live request with it.
    template <typename... Args>
    void call(const char* what, Args&&... args) const {
        py::gil_scoped_acquire acquire;
        try {
            (*fn_)(std::forward<Args>(args)...);
        } catch (const py::error_already_set& e) {
            std::cerr << "[BatchScheduler] " << what << " callback error: "
                      << e.what() << std::endl;
        }
    }

private:
    struct GilDeleter {
        void operator()(py::function* fn) const {
            py::gil_scoped_acquire acquire;
            delete fn;
        }
    };

    std::shared_ptr<py::function> fn_;
};

py::dict forward_result_dict(const ForwardResult& result) {
    py::dict out;
    out["token"] = result.token;
    out["layers"] = result.layers;
    out["dim"] = result.dim;
    out["logits"] = result.logits;
    out["top_token"] = result.top_token;
    out["correct_drafts"] = result.correct_drafts;
    out["proposed_drafts"] = result.proposed_drafts;
    out["bonus_token"] = result.bonus_token;
    out["rolled_back"] = result.rolled_back;
    out["accept_tokens"] = result.accept_tokens;
    out["accept_logits"] = result.accept_logits;
    out["accept_checksums"] = result.accept_checksums;
    out["top_logit"] = result.top_logit;
    out["checksum"] = result.checksum;
    out["position"] = result.position;
    return out;
}

py::dict mtp_stats_dict(const QwenMtpStats& stats) {
    py::dict out;
    out["verify_count"] = stats.verify_count;
    out["proposed_drafts"] = stats.proposed_drafts;
    out["correct_drafts"] = stats.correct_drafts;
    out["rollback_count"] = stats.rollback_count;
    out["replay_tokens"] = stats.replay_tokens;
    out["confidence_count"] = stats.confidence_count;
    out["confidence_sum"] = stats.confidence_sum;
    out["confidence_min"] = stats.confidence_min;
    out["confidence_max"] = stats.confidence_max;
    out["prefill_seconds"] = stats.prefill_seconds;
    out["draft_seconds"] = stats.draft_seconds;
    out["verify_seconds"] = stats.verify_seconds;
    out["replay_seconds"] = stats.replay_seconds;
    out["accept_length"] = stats.accept_length();
    out["accept_rate"] = stats.accept_rate();
    out["mean_confidence"] = stats.mean_confidence();
    return out;
}

py::dict prefix_stats_dict(const QwenPrefixCacheStats& stats) {
    py::dict out;
    out["reused_tokens"] = stats.reused_tokens;
    out["computed_tokens"] = stats.computed_tokens;
    out["prompt_tokens"] = stats.prompt_tokens;
    out["resume_source"] = stats.resume_source;
    out["matched_tokens"] = stats.matched_tokens;
    out["snapshots"] = stats.snapshots;
    out["snapshot_bytes"] = stats.snapshot_bytes;
    out["hits"] = stats.hits;
    out["misses"] = stats.misses;
    out["global_hits"] = stats.global_hits;
    out["global_misses"] = stats.global_misses;
    out["global_cached_blocks"] = stats.global_cached_blocks;
    out["global_evictions"] = stats.global_evictions;
    out["global_cache_bytes"] = stats.global_cache_bytes;
    out["global_cache_budget_bytes"] = stats.global_cache_budget_bytes;
    return out;
}

py::dict telemetry_dict(const QwenRuntimeTelemetry& telemetry) {
    py::dict out;
    py::dict checkpoint;
    checkpoint["dense_f16"] = telemetry.checkpoint_linear_kinds.dense_f16;
    checkpoint["fp8_block128"] = telemetry.checkpoint_linear_kinds.fp8_block128;
    checkpoint["fp8_channel"] = telemetry.checkpoint_linear_kinds.fp8_channel;
    checkpoint["nvfp4_group16"] = telemetry.checkpoint_linear_kinds.nvfp4_group16;
    py::dict active;
    active["dense_f16"] = telemetry.active_linear_kinds.dense_f16;
    active["fp8_block128"] = telemetry.active_linear_kinds.fp8_block128;
    active["fp8_channel"] = telemetry.active_linear_kinds.fp8_channel;
    active["nvfp4_group16"] = telemetry.active_linear_kinds.nvfp4_group16;
    out["checkpoint_linear_kinds"] = checkpoint;
    out["active_linear_kinds"] = active;
    out["nvfp4_decode_path"] = telemetry.nvfp4_decode_path;
    out["nvfp4_prefill_path"] = telemetry.nvfp4_prefill_path;
    out["fp8_channel_decode_path"] = telemetry.fp8_channel_decode_path;
    out["fp8_channel_prefill_path"] = telemetry.fp8_channel_prefill_path;
    out["target_head_path"] = telemetry.target_head_path;
    out["host_global_metadata_bytes"] = telemetry.host_global_metadata_bytes;
    out["nvfp4_q8_workspace_peak_bytes"] = telemetry.nvfp4_q8_workspace_peak_bytes;
    out["kv_paged"] = telemetry.kv_paged_blocks > 0;
    out["kv_paged_blocks"] = telemetry.kv_paged_blocks;
    out["kv_paged_block_size"] = telemetry.kv_paged_block_size;
    return out;
}

py::dict qwen_config_dict(const QwenConfig& config) {
    py::dict out;
    out["architecture"] = config.architecture;
    out["model_type"] = config.model_type;
    out["vocab_size"] = config.vocab_size;
    out["hidden_size"] = config.hidden_size;
    out["num_hidden_layers"] = config.num_hidden_layers;
    out["max_position_embeddings"] = config.max_position_embeddings;
    out["mtp_num_hidden_layers"] = config.mtp_num_hidden_layers;
    out["mtp_use_dedicated_embeddings"] = config.mtp_use_dedicated_embeddings;
    out["rms_norm_eps"] = config.rms_norm_eps;
    out["rope_theta"] = config.rope_theta;
    out["partial_rotary_factor"] = config.partial_rotary_factor;
    out["fp8_block_size"] = config.fp8_block_size;
    out["linear_attention_layers"] = config.linear_attention_layers();
    out["full_attention_layers"] = config.full_attention_layers();
    out["layer_types"] = py::list();
    for (uint64_t layer = 0; layer < config.layer_types.size(); ++layer) {
        out["layer_types"].cast<py::list>().append(config.layer_type_name(layer));
    }
    out["to_string"] = config.to_string();
    return out;
}

py::dict options_dict(const ForwardSmokeOptions& options) {
    py::dict out;
    out["tp_world"] = options.tp_world;
    out["tp_rank"] = options.tp_rank;
    out["device"] = options.device;
    out["skip_fp4_host_prepare"] = options.skip_fp4_host_prepare;
    out["nccl_id_path"] = options.nccl_id_path;
    return out;
}

ForwardResult qwen_prefill(QwenEngine& engine, const std::vector<int>& tokens,
                               int slot_id) {
    py::gil_scoped_release release;
    return engine.prefill(tokens, slot_id);
}

ForwardResult qwen_decode(QwenEngine& engine, int token, int slot_id) {
    py::gil_scoped_release release;
    return engine.decode_step(token, slot_id);
}

std::vector<ForwardResult> qwen_generate(QwenEngine& engine,
                                             const std::vector<int>& tokens,
                                             int max_new_tokens) {
    py::gil_scoped_release release;
    return engine.generate(tokens, max_new_tokens);
}

int persistent_prefill(PersistentEngine& engine, const std::vector<int>& tokens,
                       const SamplingParams& params) {
    py::gil_scoped_release release;
    return engine.prefill(tokens, params);
}

int persistent_decode(PersistentEngine& engine, int token, int position,
                      const SamplingParams& params) {
    py::gil_scoped_release release;
    return engine.decode_step(token, position, params);
}

std::vector<int> persistent_verify(PersistentEngine& engine,
                                   const std::vector<int>& tokens, int position,
                                   const SamplingParams& params) {
    py::gil_scoped_release release;
    return engine.verify_step(tokens, position, params);
}

std::vector<int> persistent_batch_verify(PersistentEngine& engine,
                                         const std::vector<int>& tokens, int position,
                                         const SamplingParams& params) {
    py::gil_scoped_release release;
    return engine.batch_verify_step(tokens, position, params);
}

}  // namespace

PYBIND11_MODULE(pocketllm_cpp, module) {
    module.doc() = "Optional native PocketLLM C++ backend";

    // Same registry the C++ CLI dispatches on, so a checkpoint reaches the same
    // engine whichever side opens it. Registration is idempotent.
    register_builtin_engines();

    // Model selection. The Python backend constructs the concrete engine classes
    // below directly, because it needs their full option surface, but it asks
    // this which one a checkpoint wants rather than carrying its own rules.
    module.def("detect_architecture", &detect_architecture,
               "Architecture a checkpoint declares, normalized to a registry key");
    module.def("registered_architectures", &registered_architectures,
               "Architectures this build has an engine for, in sorted order");

    py::enum_<QwenKvCacheDType>(module, "QwenKvCacheDType")
        .value("Fp16", QwenKvCacheDType::Fp16)
        .value("Fp8", QwenKvCacheDType::Fp8)
        .value("TurboQuantK8V4", QwenKvCacheDType::TurboQuantK8V4)
        .value("Int8PerTokenHead", QwenKvCacheDType::Int8PerTokenHead)
        .export_values();

    py::enum_<NvFp4Mode>(module, "NvFp4Mode")
        .value("Auto", NvFp4Mode::Auto)
        .value("Dp4a", NvFp4Mode::Dp4a)
        .value("Wmma", NvFp4Mode::Wmma)
        .value("Reference", NvFp4Mode::Reference)
        .export_values();

    py::enum_<OptionalSwitch>(module, "OptionalSwitch")
        .value("Auto", OptionalSwitch::Auto)
        .value("Disabled", OptionalSwitch::Disabled)
        .value("Enabled", OptionalSwitch::Enabled)
        .export_values();

    module.def("qwen_kv_cache_dtype_name", [](QwenKvCacheDType dtype) {
        return std::string(qwen_kv_cache_dtype_name(dtype));
    });
    module.def("parse_qwen_kv_cache_dtype", &parse_qwen_kv_cache_dtype);
    module.def("is_qwen3_5_checkpoint", &is_qwen3_5_checkpoint);

    py::class_<QwenKernelOptions>(module, "QwenKernelOptions")
        .def(py::init<>())
        .def_readwrite("nvfp4_mode", &QwenKernelOptions::nvfp4_mode)
        .def_readwrite("nvfp4_wide_n64", &QwenKernelOptions::nvfp4_wide_n64)
        .def_readwrite("nvfp4_wide_n64_min_rows", &QwenKernelOptions::nvfp4_wide_n64_min_rows)
        .def_readwrite("nvfp4_fused_swiglu", &QwenKernelOptions::nvfp4_fused_swiglu)
        .def_readwrite("nvfp4_shared_q8_swiglu", &QwenKernelOptions::nvfp4_shared_q8_swiglu)
        .def_readwrite("gqa_optimized", &QwenKernelOptions::gqa_optimized)
        .def_readwrite("gqa_verify_cublas_qk", &QwenKernelOptions::gqa_verify_cublas_qk)
        .def_readwrite("gqa_verify_split", &QwenKernelOptions::gqa_verify_split)
        .def_readwrite("gqa_verify_splits", &QwenKernelOptions::gqa_verify_splits)
        .def_readwrite("verify_small_fp16_cublas", &QwenKernelOptions::verify_small_fp16_cublas)
        .def_readwrite("gated_delta_flashqla", &QwenKernelOptions::gated_delta_flashqla)
        .def_readwrite("fuse_qkvz_decode", &QwenKernelOptions::fuse_qkvz_decode)
        .def_readwrite("fuse_ab_projection", &QwenKernelOptions::fuse_ab_projection)
        .def_readwrite("gated_delta_prenormalize", &QwenKernelOptions::gated_delta_prenormalize)
        .def_readwrite("gated_delta_shared_state", &QwenKernelOptions::gated_delta_shared_state)
        .def_readwrite("fuse_full_qkv_decode", &QwenKernelOptions::fuse_full_qkv_decode)
        .def_readwrite("fuse_attention_residual_norm", &QwenKernelOptions::fuse_attention_residual_norm)
        .def_readwrite("comm_overlap_slices", &QwenKernelOptions::comm_overlap_slices)
        .def("load_from_env", &QwenKernelOptions::load_from_env);

    py::class_<QwenEngineOptions>(module, "QwenEngineOptions")
        .def(py::init<>())
        .def_readwrite("tp_world", &QwenEngineOptions::tp_world)
        .def_readwrite("tp_rank", &QwenEngineOptions::tp_rank)
        .def_readwrite("device", &QwenEngineOptions::device)
        .def_readwrite("max_batch_size", &QwenEngineOptions::max_batch_size)
        .def_readwrite("prefill_chunk_tokens", &QwenEngineOptions::prefill_chunk_tokens)
        .def_readwrite("kv_cache_dtype", &QwenEngineOptions::kv_cache_dtype)
        .def_readwrite("attention_window", &QwenEngineOptions::attention_window)
        .def_readwrite("attention_sink_tokens", &QwenEngineOptions::attention_sink_tokens)
        .def_readwrite("prefix_cache", &QwenEngineOptions::prefix_cache)
        .def_readwrite("state_snapshot_interval_tokens", &QwenEngineOptions::state_snapshot_interval_tokens)
        .def_readwrite("max_state_snapshots", &QwenEngineOptions::max_state_snapshots)
        .def_readwrite("mtp", &QwenEngineOptions::mtp)
        .def_readwrite("mtp_speculative_tokens", &QwenEngineOptions::mtp_speculative_tokens)
        .def_readwrite("mtp_adaptive", &QwenEngineOptions::mtp_adaptive)
        .def_readwrite("dspark_checkpoint", &QwenEngineOptions::dspark_checkpoint)
        .def_readwrite("dflash2_checkpoint", &QwenEngineOptions::dflash2_checkpoint)
        .def_readwrite("nccl_id_path", &QwenEngineOptions::nccl_id_path)
        .def_readwrite("temperature", &QwenEngineOptions::temperature)
        .def_readwrite("top_p", &QwenEngineOptions::top_p)
        .def_readwrite("top_k", &QwenEngineOptions::top_k)
        .def_readwrite("sampling_seed", &QwenEngineOptions::sampling_seed)
        .def_readwrite("kv_paged", &QwenEngineOptions::kv_paged)
        .def_readwrite("kv_block_size", &QwenEngineOptions::kv_block_size)
        .def_readwrite("kv_cache_bytes", &QwenEngineOptions::kv_cache_bytes)
        .def_readwrite("prefix_cache_bytes", &QwenEngineOptions::prefix_cache_bytes)
        .def_readwrite("kernel", &QwenEngineOptions::kernel);

    py::class_<ForwardResult>(module, "QwenForwardResult")
        .def(py::init<>())
        .def_readwrite("token", &ForwardResult::token)
        .def_readwrite("layers", &ForwardResult::layers)
        .def_readwrite("dim", &ForwardResult::dim)
        .def_readwrite("logits", &ForwardResult::logits)
        .def_readwrite("top_token", &ForwardResult::top_token)
        .def_readwrite("correct_drafts", &ForwardResult::correct_drafts)
        .def_readwrite("proposed_drafts", &ForwardResult::proposed_drafts)
        .def_readwrite("bonus_token", &ForwardResult::bonus_token)
        .def_readwrite("rolled_back", &ForwardResult::rolled_back)
        .def_readwrite("accept_tokens", &ForwardResult::accept_tokens)
        .def_readwrite("accept_logits", &ForwardResult::accept_logits)
        .def_readwrite("accept_checksums", &ForwardResult::accept_checksums)
        .def_readwrite("top_logit", &ForwardResult::top_logit)
        .def_readwrite("checksum", &ForwardResult::checksum)
        .def_readwrite("position", &ForwardResult::position)
        .def("as_dict", &forward_result_dict);

    py::class_<QwenMtpStats>(module, "QwenMtpStats")
        .def(py::init<>())
        .def_readwrite("verify_count", &QwenMtpStats::verify_count)
        .def_readwrite("proposed_drafts", &QwenMtpStats::proposed_drafts)
        .def_readwrite("correct_drafts", &QwenMtpStats::correct_drafts)
        .def_readwrite("rollback_count", &QwenMtpStats::rollback_count)
        .def_readwrite("replay_tokens", &QwenMtpStats::replay_tokens)
        .def_readwrite("confidence_count", &QwenMtpStats::confidence_count)
        .def_readwrite("confidence_sum", &QwenMtpStats::confidence_sum)
        .def_readwrite("confidence_min", &QwenMtpStats::confidence_min)
        .def_readwrite("confidence_max", &QwenMtpStats::confidence_max)
        .def_readwrite("prefill_seconds", &QwenMtpStats::prefill_seconds)
        .def_readwrite("draft_seconds", &QwenMtpStats::draft_seconds)
        .def_readwrite("verify_seconds", &QwenMtpStats::verify_seconds)
        .def_readwrite("replay_seconds", &QwenMtpStats::replay_seconds)
        .def_property_readonly("accept_length", &QwenMtpStats::accept_length)
        .def_property_readonly("accept_rate", &QwenMtpStats::accept_rate)
        .def_property_readonly("mean_confidence", &QwenMtpStats::mean_confidence)
        .def("as_dict", &mtp_stats_dict);

    py::class_<QwenPrefixCacheStats>(module, "QwenPrefixCacheStats")
        .def(py::init<>())
        .def_readwrite("reused_tokens", &QwenPrefixCacheStats::reused_tokens)
        .def_readwrite("computed_tokens", &QwenPrefixCacheStats::computed_tokens)
        .def_readwrite("prompt_tokens", &QwenPrefixCacheStats::prompt_tokens)
        .def_readwrite("resume_source", &QwenPrefixCacheStats::resume_source)
        .def_readwrite("matched_tokens", &QwenPrefixCacheStats::matched_tokens)
        .def_readwrite("snapshots", &QwenPrefixCacheStats::snapshots)
        .def_readwrite("snapshot_bytes", &QwenPrefixCacheStats::snapshot_bytes)
        .def_readwrite("hits", &QwenPrefixCacheStats::hits)
        .def_readwrite("misses", &QwenPrefixCacheStats::misses)
        .def_readwrite("global_hits", &QwenPrefixCacheStats::global_hits)
        .def_readwrite("global_misses", &QwenPrefixCacheStats::global_misses)
        .def_readwrite("global_cached_blocks", &QwenPrefixCacheStats::global_cached_blocks)
        .def_readwrite("global_evictions", &QwenPrefixCacheStats::global_evictions)
        .def_readwrite("global_cache_bytes", &QwenPrefixCacheStats::global_cache_bytes)
        .def_readwrite("global_cache_budget_bytes", &QwenPrefixCacheStats::global_cache_budget_bytes)
        .def("as_dict", &prefix_stats_dict);

    py::class_<QwenEngine>(module, "QwenEngine")
        .def(py::init<const std::string&, const QwenEngineOptions&, int, int>(),
             py::arg("checkpoint_dir"), py::arg("options"),
             py::arg("layer_count") = 0, py::arg("max_context") = 8192,
             "Construct the stateful native Qwen engine.")
        .def("warmup_tp", [](QwenEngine& engine) {
            py::gil_scoped_release release;
            engine.warmup_tp();
        })
        // reset()/clear_prefix_cache() zero the recurrent state on the device, so
        // they release the GIL like the other device-touching entry points.
        .def("reset", [](QwenEngine& engine) {
            py::gil_scoped_release release;
            engine.reset();
        })
        .def("clear_prefix_cache", [](QwenEngine& engine) {
            py::gil_scoped_release release;
            engine.clear_prefix_cache();
        })
        .def("prefill", &qwen_prefill, py::arg("token_ids"), py::arg("slot_id") = 0)
        .def("decode_step", &qwen_decode, py::arg("token_id"), py::arg("slot_id") = 0)
        .def("generate", &qwen_generate)
        .def("run_worker_loop", [](QwenEngine& engine) {
            py::gil_scoped_release release;
            engine.run_worker_loop();
        }, "TP rank > 0 entry point: blocks on NCCL command channel until shutdown")
        .def("worker_command_prefill", [](QwenEngine& engine, const std::vector<int>& token_ids,
                                          int32_t slot_id) {
            py::gil_scoped_release release;
            engine.worker_command_prefill(token_ids, slot_id);
        }, py::arg("token_ids"), py::arg("slot_id") = 0)
        .def("worker_command_decode", [](QwenEngine& engine, int32_t last_token,
                                         int32_t slot_id) {
            py::gil_scoped_release release;
            engine.worker_command_decode(last_token, slot_id);
        }, py::arg("last_token"), py::arg("slot_id") = 0)
        .def("worker_command_reset", [](QwenEngine& engine) {
            py::gil_scoped_release release;
            engine.worker_command_reset();
        })
        .def("worker_command_shutdown", [](QwenEngine& engine) {
            py::gil_scoped_release release;
            engine.worker_command_shutdown();
        })
        .def_property_readonly("position", &QwenEngine::position)
        .def_property_readonly("max_context", &QwenEngine::max_context)
        .def_property_readonly("resident_weight_bytes", &QwenEngine::resident_weight_bytes)
        .def_property_readonly("resident_scale_bytes", &QwenEngine::resident_scale_bytes)
        .def_property_readonly("verify_weight_bytes", &QwenEngine::verify_weight_bytes)
        .def_property_readonly("activation_workspace_peak_bytes", &QwenEngine::activation_workspace_peak_bytes)
        .def_property_readonly("kv_cache_bytes", &QwenEngine::kv_cache_bytes)
        .def_property_readonly("kv_cache_scale_bytes", &QwenEngine::kv_cache_scale_bytes)
        .def_property_readonly("kv_paged", &QwenEngine::kv_paged)
        .def_property_readonly("kv_free_blocks", &QwenEngine::kv_free_blocks)
        .def_property_readonly("kv_total_blocks", &QwenEngine::kv_total_blocks)
        .def_property_readonly("kv_cache_pinned_blocks", &QwenEngine::kv_cache_pinned_blocks)
        .def_property_readonly("config", [](const QwenEngine& engine) {
            return qwen_config_dict(engine.config());
        })
        .def_property_readonly("prefix_cache_stats", [](const QwenEngine& engine) {
            return prefix_stats_dict(engine.prefix_cache_stats());
        })
        .def_property_readonly("mtp_stats", [](const QwenEngine& engine) {
            return mtp_stats_dict(engine.mtp_stats());
        })
        .def_property_readonly("runtime_telemetry", [](const QwenEngine& engine) {
            return telemetry_dict(engine.runtime_telemetry());
        });

    py::class_<SamplingParams>(module, "SamplingParams")
        .def(py::init<>())
        .def_readwrite("temperature", &SamplingParams::temperature)
        .def_readwrite("top_p", &SamplingParams::top_p)
        .def_readwrite("greedy", &SamplingParams::greedy)
        .def_readwrite("seed", &SamplingParams::seed);

    py::class_<ForwardSmokeOptions>(module, "ForwardSmokeOptions")
        .def(py::init<>())
        .def_readwrite("tp_world", &ForwardSmokeOptions::tp_world)
        .def_readwrite("tp_rank", &ForwardSmokeOptions::tp_rank)
        .def_readwrite("device", &ForwardSmokeOptions::device)
        .def_readwrite("skip_fp4_host_prepare", &ForwardSmokeOptions::skip_fp4_host_prepare)
        .def_readwrite("nccl_id_path", &ForwardSmokeOptions::nccl_id_path)
        .def("as_dict", &options_dict);

    py::class_<PersistentEngine>(module, "PersistentEngine")
        .def(py::init<const std::string&, const ForwardSmokeOptions&, int, int>(),
             py::arg("checkpoint_dir"), py::arg("options"),
             py::arg("layer_count"), py::arg("max_context"))
        .def("reset_session", [](PersistentEngine& engine) {
            py::gil_scoped_release release;
            engine.reset_session();
        })
        .def("prefill", &persistent_prefill)
        .def("decode_step", &persistent_decode)
        .def("verify_step", &persistent_verify)
        .def("batch_verify_step", &persistent_batch_verify)
        .def("speculative_step", [](PersistentEngine& engine, int token, int position,
                                     const SamplingParams& params) {
            py::gil_scoped_release release;
            return engine.speculative_step(token, position, params);
        })
        .def("load_dspark", [](PersistentEngine& engine, const std::string& path) {
            py::gil_scoped_release release;
            engine.load_dspark(path);
        })
        .def("dspark_loaded", &PersistentEngine::dspark_loaded)
        .def("eos_id", &PersistentEngine::eos_id)
        .def("batched_decode_enabled", &PersistentEngine::batched_decode_enabled)
        .def("max_context", &PersistentEngine::max_context)
        .def("layer_count", &PersistentEngine::layer_count)
        .def_property_readonly("last_dspark_hidden", &PersistentEngine::last_dspark_hidden)
        .def_property_readonly("last_dspark_hidden_positions", &PersistentEngine::last_dspark_hidden_positions)
        .def_property_readonly("last_verify_dspark_hidden", &PersistentEngine::last_verify_dspark_hidden)
        .def_property_readonly("last_topk_tokens", &PersistentEngine::last_topk_tokens)
        .def_property_readonly("last_topk_logits", &PersistentEngine::last_topk_logits)
        .def("warmup_tp", [](PersistentEngine& engine) {
            py::gil_scoped_release release;
            engine.warmup_tp();
        })
        .def("run_worker_loop", [](PersistentEngine& engine) {
            py::gil_scoped_release release;
            engine.run_worker_loop();
        })
        .def("worker_command_prefill", &PersistentEngine::worker_command_prefill,
             py::arg("token_ids"), py::arg("slot_id") = 0)
        .def("worker_command_decode", &PersistentEngine::worker_command_decode,
             py::arg("last_token"), py::arg("position"), py::arg("slot_id") = 0)
        .def("worker_command_reset", &PersistentEngine::worker_command_reset)
        .def("worker_command_reset_slot", &PersistentEngine::worker_command_reset_slot,
             py::arg("slot_id"))
        .def("worker_command_shutdown", &PersistentEngine::worker_command_shutdown)
        .def("worker_command_verify", &PersistentEngine::worker_command_verify)
        .def("worker_command_batch_verify", &PersistentEngine::worker_command_batch_verify)
        .def("worker_command_finalize_batch_verify", &PersistentEngine::worker_command_finalize_batch_verify)
        .def("worker_command_draft", &PersistentEngine::worker_command_draft)
        .def("worker_command_speculative_decode", &PersistentEngine::worker_command_speculative_decode)
        .def("worker_command_prime_draft_kv", &PersistentEngine::worker_command_prime_draft_kv);

    // Capabilities struct for engine introspection
    py::class_<Capabilities>(module, "Capabilities")
        .def(py::init<>())
        .def_readonly("paged_kv", &Capabilities::paged_kv)
        .def_readonly("continuous_batching", &Capabilities::continuous_batching)
        .def_readonly("chunked_prefill", &Capabilities::chunked_prefill)
        .def_readonly("max_slots", &Capabilities::max_slots)
        .def_readonly("per_request_sampling", &Capabilities::per_request_sampling)
        .def_readonly("per_request_top_k", &Capabilities::per_request_top_k)
        .def_readonly("fixed_temperature", &Capabilities::fixed_temperature)
        .def_readonly("fixed_top_p", &Capabilities::fixed_top_p)
        .def_readonly("fixed_top_k", &Capabilities::fixed_top_k)
        .def_readonly("fixed_seed", &Capabilities::fixed_seed)
        .def("__repr__", [](const Capabilities& c) {
            return "<Capabilities max_slots=" + std::to_string(c.max_slots) +
                   " continuous_batching=" + (c.continuous_batching ? "True" : "False") +
                   " chunked_prefill=" + (c.chunked_prefill ? "True" : "False") +
                   " paged_kv=" + (c.paged_kv ? "True" : "False") + ">";
        });

    // BatchScheduler bindings (Phase 3.4)
    py::class_<BatchSamplingParams>(module, "QwenBatchSamplingParams")
        .def(py::init<>())
        .def_readwrite("temperature", &BatchSamplingParams::temperature)
        .def_readwrite("top_p", &BatchSamplingParams::top_p)
        .def_readwrite("top_k", &BatchSamplingParams::top_k)
        .def_readwrite("seed", &BatchSamplingParams::seed)
        .def_readwrite("max_new_tokens", &BatchSamplingParams::max_new_tokens)
        // Exposed so a Python benchmark can keep its token counts fixed
        // (ignore_eos) or stop on its own tokenizer's ids (stop_token_ids)
        // instead of the checkpoint's.
        .def_readwrite("stop_token_ids", &BatchSamplingParams::stop_token_ids)
        .def_readwrite("ignore_eos", &BatchSamplingParams::ignore_eos);

    py::class_<SchedulerGenerationResult>(module, "SchedulerGenerationResult")
        .def(py::init<>())
        .def_readwrite("request_id", &SchedulerGenerationResult::request_id)
        .def_readwrite("generated_tokens", &SchedulerGenerationResult::generated_tokens)
        .def_readwrite("finish_reason", &SchedulerGenerationResult::finish_reason)
        .def_readwrite("constraint_completed", &SchedulerGenerationResult::constraint_completed)
        .def_readwrite("prompt_tokens", &SchedulerGenerationResult::prompt_tokens)
        .def_readwrite("completion_tokens", &SchedulerGenerationResult::completion_tokens)
        .def_readwrite("proposed_drafts", &SchedulerGenerationResult::proposed_drafts)
        .def_readwrite("accepted_drafts", &SchedulerGenerationResult::accepted_drafts)
        .def_readwrite("speculative_steps", &SchedulerGenerationResult::speculative_steps)
        .def_readwrite("rollback_steps", &SchedulerGenerationResult::rollback_steps)
        .def_readwrite("total_seconds", &SchedulerGenerationResult::total_seconds)
        .def_readwrite("ttft_seconds", &SchedulerGenerationResult::ttft_seconds)
        // The same split the server records into its Prometheus histograms, so a
        // Python caller can attribute a request without scraping /metrics. Each
        // is negative when the interval has no second endpoint: a request that
        // was never admitted has no queue wait, and one that produced no token
        // has neither a prefill nor a decode interval.
        .def_readwrite("queue_seconds", &SchedulerGenerationResult::queue_seconds)
        .def_readwrite("prefill_seconds", &SchedulerGenerationResult::prefill_seconds)
        .def_readwrite("decode_seconds", &SchedulerGenerationResult::decode_seconds)
        .def_readwrite("error", &SchedulerGenerationResult::error);

    py::class_<BatchScheduler::Stats>(module, "QwenBatchSchedulerStats")
        .def(py::init<>())
        .def_readwrite("waiting_requests", &BatchScheduler::Stats::waiting_requests)
        .def_readwrite("running_requests", &BatchScheduler::Stats::running_requests)
        .def_readwrite("completed_requests", &BatchScheduler::Stats::completed_requests)
        .def_readwrite("cancelled_requests", &BatchScheduler::Stats::cancelled_requests)
        .def_readwrite("free_slots", &BatchScheduler::Stats::free_slots)
        .def_readwrite("cache_pinned_blocks", &BatchScheduler::Stats::cache_pinned_blocks);

    py::class_<BatchScheduler>(module, "QwenBatchScheduler")
        .def(py::init<QwenEngine*, int>(),
             py::arg("engine"), py::arg("max_batch_size"))
        .def("submit_request",
             [](BatchScheduler& scheduler,
                const std::vector<int>& prompt_tokens,
                const BatchSamplingParams& sampling,
                py::object callback,
                py::object on_token) -> uint64_t {

                 std::function<void(const SchedulerGenerationResult&)> cpp_callback;
                 if (!callback.is_none()) {
                     GilSafeCallable fn(py::cast<py::function>(callback));
                     cpp_callback = [fn](const SchedulerGenerationResult& result) {
                         fn.call("completion", result);
                     };
                 }

                 TokenCallback cpp_token_callback;
                 if (!on_token.is_none()) {
                     GilSafeCallable fn(py::cast<py::function>(on_token));
                     cpp_token_callback = [fn](uint64_t request_id, int token) {
                         fn.call("token", request_id, token);
                     };
                 }

                 // Moved, not copied: copying a std::function that owns a
                 // py::function touches the Python refcount, and the GIL is
                 // released below. Moving is a pointer swap, so it is safe.
                 py::gil_scoped_release release;
                 return scheduler.submit_request(prompt_tokens, sampling,
                                                std::move(cpp_callback),
                                                std::move(cpp_token_callback));
             },
             py::arg("prompt_tokens"),
             py::arg("sampling"),
             py::arg("callback") = py::none(),
             py::arg("on_token") = py::none(),
             R"doc(Submit a generation request.

             Args:
                 prompt_tokens: Input token IDs
                 sampling: Sampling parameters (temperature, top_p, etc.)
                 callback: Optional completion callback (called from scheduler thread)
                 on_token: Optional per-token callback (called from scheduler thread)

             Returns:
                 request_id (> 0 on success, 0 on failure)

             Note:
                 Callbacks run from the scheduler's background thread and MUST NOT BLOCK.
                 Blocking callbacks will stall all running requests.
             )doc")
        .def("cancel_request", [](BatchScheduler& scheduler, uint64_t request_id) {
            py::gil_scoped_release release;
            return scheduler.cancel_request(request_id);
        }, py::arg("request_id"))
        .def("poll_result", [](BatchScheduler& scheduler, uint64_t request_id, int timeout_ms) -> py::object {
            SchedulerGenerationResult result;
            bool success;
            {
                py::gil_scoped_release release;
                success = scheduler.poll_result(request_id, &result, timeout_ms);
            }
            if (success) {
                return py::cast(result);
            } else {
                return py::none();
            }
        }, py::arg("request_id"), py::arg("timeout_ms") = 30000)
        .def("get_stats", [](BatchScheduler& scheduler) {
            return scheduler.get_stats();
        })
        .def("is_running", &BatchScheduler::is_running)
        .def("stop", [](BatchScheduler& scheduler) {
            py::gil_scoped_release release;
            scheduler.stop();
        })
        .def("engine_caps", &BatchScheduler::engine_caps,
             py::return_value_policy::reference_internal,
             "Query engine capabilities (max_slots, continuous_batching, etc.)")
        .def("max_batch_size", &BatchScheduler::max_batch_size,
             "Get effective batch size (may be clamped to max_slots)")
        .def("set_prefill_token_budget", &BatchScheduler::set_prefill_token_budget,
             py::arg("tokens"),
             R"doc(Set prefill token budget per schedule iteration.

             Smaller values let decode interleave sooner at some cost to prefill
             throughput. 0 disables chunking (run each prompt to completion).

             Ignored if engine does not declare chunked_prefill capability.
             )doc")
        .def("prefill_token_budget", &BatchScheduler::prefill_token_budget,
             "Get current prefill token budget");

    module.attr("backend") = device_backend_name();
}
