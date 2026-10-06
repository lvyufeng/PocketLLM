/* A persistent thread pool, and the one entry point the kernels share.
 *
 * The graph walks roughly two hundred kernel calls per token, so a pool that
 * spawned and joined threads per call would spend more time in the runtime than
 * in the arithmetic.  The pool is created once, on first use, and reused for
 * every call for the life of the process.
 *
 * **Not OpenMP.**  `-fopenmp` would make `libgomp` a runtime dependency of
 * libpocketllm.so, and this library's whole reason to exist is an edge and
 * mobile target where the fewer things the loader has to find the better.  A
 * `std::thread` pool needs nothing beyond libstdc++ and pthread, which are
 * already there.
 *
 * ## The one rule
 *
 * **Parallelise only over an axis whose outputs are independent.  Never split a
 * reduction.**  Every call site in `kernel/kernels.cpp` splits an outer index
 * whose iterations touch disjoint memory -- the output column of a GEMM, the
 * token of a gather, the element of an elementwise op.  Splitting the `k` axis
 * of a dot product and summing the partials would be faster still and is
 * forbidden: it changes the association order, so the result moves with the
 * thread count and stops being reproducible.  `gemm_quant`'s comment says the
 * same thing where someone would be tempted to try it.
 *
 * Because the partition is a fixed contiguous split of the outer index and the
 * arithmetic inside each range is untouched, the output of a parallelised
 * kernel is *bit-identical* to the single-threaded one.  That is the property
 * `tests/native/test_cpu_parallel.py` checks by running each op at one thread
 * and at eight and diffing the bytes.
 */

#ifndef POCKETLLM_KERNEL_PARALLEL_H
#define POCKETLLM_KERNEL_PARALLEL_H

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define POCKETLLM_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define POCKETLLM_PAUSE() __builtin_arm_yield()
#else
#define POCKETLLM_PAUSE() ((void)0)
#endif

namespace pocketllm {
namespace kernel {

/* The machine's physical core count, or 0 when it cannot be read.
 *
 * Linux exposes the topology in `sysfs`: one `topology/thread_siblings_list`
 * per logical CPU lists that CPU's siblings, so the distinct lists are the
 * cores.  This is deliberately *not* `hardware_concurrency`, which counts
 * logical CPUs -- and on this host that is the difference between 44 and 88.
 *
 * The reason the distinction matters is specific to how this pool works: the
 * workers spin, so every thread the pool holds is at 100% for the duration of a
 * job.  Two hyperthread siblings that both spin are fighting over one core's
 * issue ports and one core's share of the memory pipeline, and the cost shows
 * up in the arithmetic they are supposed to be doing.  Measured on
 * `qwen3-0.6b-q4_k_m.gguf` at the shipped default, threads against throughput:
 *
 *     threads   pp512   tg64
 *        22      521     74.2      one socket's cores
 *        44      939     72.4      both sockets' cores   <- physical
 *        88      862     54.5      every hardware thread <- hardware_concurrency
 *
 * Prefill is 9% faster at 44 than at 88 and decode is 33% faster, so the
 * default this function returns is the number of *cores*, not the number of
 * CPUs. */
inline int64_t physical_core_count() {
#if defined(__linux__)
  DIR *dir = opendir("/sys/devices/system/cpu");
  if (dir == nullptr) {
    return 0;
  }
  std::set<int> seen;
  int cores = 0;
  struct dirent *entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    if (std::strncmp(entry->d_name, "cpu", 3) != 0 || !std::isdigit(entry->d_name[3])) {
      continue;
    }
    /* `thread_siblings_list` is this CPU's sibling set as a ranges list such as
     * `0-21,44-65`; the first number is enough to identify the core, because
     * every CPU in one core's file names the same set. */
    const std::string path = std::string("/sys/devices/system/cpu/") + entry->d_name +
                             "/topology/thread_siblings_list";
    FILE *file = std::fopen(path.c_str(), "r");
    if (file == nullptr) {
      continue;
    }
    int first = 0;
    const int scanned = std::fscanf(file, "%d", &first);
    std::fclose(file);
    if (scanned == 1 && seen.insert(first).second) {
      ++cores;
    }
  }
  closedir(dir);
  return cores;
#else
  return 0;
#endif
}

/* The distinct NUMA nodes this process can run on, or 0 when it cannot be read.
 *
 * `node*` directories under `/sys/devices/system/node` are the nodes that have
 * memory attached; a container sees the ones it is allowed.  This is not
 * `numactl --hardware`: the tree is read directly because the pool cannot
 * shell out and `libnuma` is a dependency this library exists to avoid. */
inline int64_t numa_node_count() {
#if defined(__linux__)
  DIR *dir = opendir("/sys/devices/system/node");
  if (dir == nullptr) {
    return 0;
  }
  int nodes = 0;
  struct dirent *entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    if (std::strncmp(entry->d_name, "node", 4) == 0 && std::isdigit(entry->d_name[4])) {
      ++nodes;
    }
  }
  closedir(dir);
  return nodes;
#else
  return 0;
#endif
}

/* How many threads the CPU kernels should use.
 *
 * `$POCKETLLM_CPU_THREADS` wins when it is set and parses to a positive number,
 * which is the escape hatch for reproducing a single-threaded number
 * (``POCKETLLM_CPU_THREADS=1``) and for staying a good citizen on a shared
 * host.
 *
 * Otherwise it is the physical cores of **one NUMA node**, not the whole
 * machine, and that default is a measurement rather than a caution.  Using
 * every core means a thread on each node and the weights reachable from one of
 * them; measured on the 2 x 22-core host with `pp512`/`tg64`, threads against
 * throughput:
 *
 *     threads   22 (one node)   44 (both)
 *     pp512          579-583      735-849
 *     tg64            85-87        69-73
 *
 * Decode is **1.12-1.26x faster on one node** across paired runs while prefill
 * is ~20% faster on both, because decode is latency-bound on a weight stream the
 * remote node reaches across the interconnect and prefill has the arithmetic to
 * hide it.  Decode is the number the edge target lives on -- time per token -- so
 * the default is the node, and a caller who wants the last prefill percent can
 * set `POCKETLLM_CPU_THREADS` to the core count.  A single-node machine (which is
 * every phone and most edge boards) is unaffected: `cores == node_cores`.
 *
 * Clamped to [1, 256] so a typo cannot start a thousand threads. */
inline int64_t cpu_thread_count() {
  const char *from_env = std::getenv("POCKETLLM_CPU_THREADS");
  if (from_env != nullptr) {
    char *end = nullptr;
    const long parsed = std::strtol(from_env, &end, 10);
    if (end != from_env && parsed > 0) {
      return parsed > 256 ? 256 : static_cast<int64_t>(parsed);
    }
  }
  const int64_t cores = physical_core_count();
  const int64_t nodes = numa_node_count();
  /* A multi-node machine gets one node's share of the cores.  The count is
   * cores-per-node rather than a core *mask* on purpose: the pool already lets
   * the scheduler place its threads, and pinning would be a second policy to get
   * wrong (and the wrong one on a machine where another process owns a node).
   * Halving is exact for the symmetric topology this is for; an asymmetric one
   * would want the per-node lists and is not what the default is optimizing. */
  int64_t want = (nodes > 1 && cores > 0) ? (cores + nodes - 1) / nodes : cores;
  if (want <= 0) {
    const unsigned hardware = std::thread::hardware_concurrency();
    want = hardware == 0 ? 1 : static_cast<int64_t>(hardware);
  }
  return want > 256 ? 256 : want;
}

namespace detail {

/* How many contiguous ranges a job of `total` units splits into, given a pool
 * of `threads` and a `min_per_task` floor.
 *
 * Shared between the pool that runs the ranges and `parallel_tasks`, which a
 * caller uses to size per-task scratch; the two must agree or a buffer is sized
 * for a partition that does not happen.  It takes the thread count as an
 * argument so it is callable before the pool exists. */
/* Which chunk this thread runs.  Zero everywhere by default, which is the
 * caller's slot; a background worker writes its own index once, in
 * `worker_main`, and never again.  `thread_local` is what lets `run_ranges`
 * hand out a chunk by index without an atomic claim -- see `run_worker_job`. */
inline thread_local int64_t t_chunk_index = 0;

inline int64_t partition_size(int64_t total, int64_t min_per_task, int64_t threads) {
  if (total <= 0 || threads <= 0) {
    return 0;
  }
  int64_t chunks = (total + min_per_task - 1) / min_per_task;
  if (chunks < 1) {
    chunks = 1;
  }
  return chunks < threads ? chunks : threads;
}

/* The pool, as a singleton.
 *
 * One parallel call is in flight at a time.  That is not a limitation to work
 * around: the engine's whole design is one session driving one device from one
 * thread, so two kernels are never running at once and a kernel never nests a
 * parallel call inside another.  The mutex serialises *callers* and is held
 * across the whole call -- `run_ranges` says why it has to be -- and `job_` is
 * a reference into the caller's frame that outlives the call because the caller
 * blocks at a barrier until every worker has *left* the job -- not merely until
 * every chunk has run.  A worker between its last chunk and its next claim is
 * still reading the job, so the caller cannot return while one is in that
 * window or the next job's publication would race it.
 *
 * ## Sleeping is what it used to do, and it was the whole cost
 *
 * The first version of this pool had the workers wait on a condition variable
 * and the caller notify at the end of the job.  That is the textbook shape and
 * it is wrong for this workload by an order of magnitude.  A decode token is
 * 198 `parallel_for` calls -- 28 layers of seven GEMMs plus the embedding and
 * the head -- and the individual calls are short: at 22 threads a 1024x1024
 * projection is on the order of a hundred microseconds.  A futex round trip
 * (wait, notify, wake, return) costs roughly ten to thirty of those microseconds
 * on this host, so the pool spent most of a token waking up threads that had
 * just gone to sleep.
 *
 * The measured numbers, `tg64` on `qwen3-0.6b-q4_k_m.gguf`, threads pinned to
 * one socket, median of three:
 *
 *     condvar barrier, t=8    22.96 tok/s
 *     condvar barrier, t=22   22.56 tok/s     <- no scaling at all
 *     spin barrier,    t=8    31.23 tok/s
 *     spin barrier,    t=22   56.10 tok/s     <- 2.5x
 *
 * The flat condvar scaling is the tell: if the barrier costs the same whether
 * eight threads or twenty-two have to be woken through the same kernel object,
 * then adding threads adds no throughput and every core past the first is
 * paying for a wake-up it does not need.
 *
 * So the workers spin.  `generation_` is the only thing they read while idle,
 * and the caller's next job bumps it within microseconds -- often before the
 * worker has finished its pause loop.  The barrier at the end of the job spins
 * on `arrived_in_job_` the same way.
 *
 * The cost is stated rather than hidden: while a job is in flight, the pool's
 * cores are at 100% even when the job is tiny, where the condvar version let
 * them idle.  For a batch server sharing a machine that would be a reason not to
 * do this -- and `$POCKETLLM_CPU_THREADS` is the polite setting there.  For one
 * session driving one device from one thread, the machine is the session's and
 * the idle time was the cost of the wake-ups that are no longer happening.
 *
 * A fallback to the sleeping path on a long spin is deliberately *not* here.
 * The spin is bounded by the work remaining: the caller reaches the barrier
 * only after running its own chunk, so the longest a worker can still be busy
 * is one chunk, which is a fixed fraction of a job.  There is no unbounded wait
 * for a fallback to protect against -- if a worker is descheduled mid-chunk the
 * caller waits for it either way, and the version that sleeps would too.
 */
class ThreadPool {
 public:
  static ThreadPool &instance() {
    static ThreadPool pool;
    return pool;
  }

  int64_t size() const { return workers_.empty() ? 1 : static_cast<int64_t>(workers_.size()) + 1; }

  /* Run ``fn(lo, hi, chunk)`` for a set of disjoint contiguous ranges covering
   * ``[0, total)``.  Blocks until every range has been run.  ``chunk`` is the
   * range's index in ``[0, chunks)``, which a caller needs when each task owns
   * a private scratch region rather than just an output range (``attention``).
   *
   * At most ``size()`` ranges are handed out, so each one is at least
   * ``total / size()`` long and the per-range call overhead is amortised
   * whatever ``total`` is.  ``min_per_task`` is a floor on the work handed to
   * one range: when the whole job is smaller than one task the call runs
   * serially rather than paying to wake the pool. */
  template <typename Fn>
  void run_ranges(int64_t total, int64_t min_per_task, Fn &&fn) {
    if (total <= 0) {
      return;
    }
    const int64_t chunks = partition_size(total, min_per_task, size());
    if (chunks <= 1) {
      fn(static_cast<int64_t>(0), total, static_cast<int64_t>(0));
      return;
    }

    /* The lock is held across the whole call, and what it buys is worth stating
     * precisely because it is not the obvious thing.
     *
     * The barrier below already guarantees that no worker is inside the body
     * when the caller returns: a worker publishes its arrival after its claim
     * loop has run dry, so "arrived" implies "out of the job", and the caller
     * does not leave until every worker has arrived.  Holding `job_`'s captures
     * -- references into the caller's frame -- across the barrier is safe on
     * that basis alone.
     *
     * What the lock adds is that no *second* job can be published while this one
     * is outstanding.  Without it, a worker that was descheduled through an
     * entire job could wake to see the generation two ahead, join the newer job,
     * and never arrive for the older one -- and the older job's caller, which is
     * spinning on the arrival count, would wait for an arrival that can no
     * longer come.  That is a hang rather than a wrong number, and it is the
     * reason the publish and the barrier are one critical section instead of
     * two.
     *
     * The engine has one caller and one session by design, so in practice the
     * lock is uncontended; it costs two futex-free acquisitions per kernel call.
     * */
    std::lock_guard<std::mutex> lock(mutex_);
    /* A reference into the caller's frame: valid because the caller blocks at
     * the barrier below until every worker has left this job, and the caller's
     * frame cannot unwind while it is blocked there. */
    job_ = [&fn, total, chunks](int64_t chunk) {
      const int64_t lo = total * chunk / chunks;
      const int64_t hi = total * (chunk + 1) / chunks;
      fn(lo, hi, chunk);
    };
    chunks_ = chunks;
    /* Not every worker necessarily takes a chunk on every job -- one can be
     * descheduled across the whole call -- so the barrier counts arrivals at
     * the *end* of the job rather than chunks claimed.  Every worker that is
     * running tells the barrier; the caller knows the count. */
    workers_left_ = static_cast<int64_t>(workers_.size());
    arrived_in_job_.store(0, std::memory_order_relaxed);
    /* The publish is the generation bump, and it must be the *last* store of
     * the job's state: a worker that sees the new generation reads everything
     * above it, and the release ordering is what makes that true without a
     * second lock on the reader side. */
    ++generation_;

    /* This thread takes chunk zero -- see `run_worker_job` for why the chunks
     * are handed out by index rather than claimed, and why the caller runs one
     * rather than draining the queue. */
    job_(0);

    /* The barrier spins rather than sleeping -- see the class comment for the
     * measurement that decided this.  A worker publishes its arrival with
     * release ordering after its last chunk, so observing the count also
     * observes the chunk's writes. */
    while (arrived_in_job_.load(std::memory_order_acquire) != workers_left_) {
      POCKETLLM_PAUSE();
    }
  }

 private:
  ThreadPool() {
    const int64_t wanted = cpu_thread_count();
    workers_.reserve(static_cast<std::size_t>(wanted));
    /* ``wanted - 1`` background threads: the calling thread is the last
     * executor, so a pool of N uses N cores rather than N+1. */
    next_worker_index_.store(0, std::memory_order_relaxed);
    for (int64_t i = 1; i < wanted; ++i) {
      workers_.emplace_back([this] { worker_main(); });
    }
  }

  ~ThreadPool() {
    stop_.store(true, std::memory_order_relaxed);
    ++generation_;
    for (std::thread &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  /* Run this thread's chunk.  Which one is `chunk_index()`, and the point of the
   * index is that there is nothing to claim.
   *
   * The first version handed chunks out through an atomic `fetch_add` -- the
   * textbook work-stealing queue -- and it was paying for a property the pool
   * cannot use.  `partition_size` floors the chunk count at the thread count and
   * caps it there, so a job hands out *at most one chunk per thread*: the claim
   * loop could never run twice for any worker, and stealing was impossible.  All
   * the atomic did was put every worker on one cache line at the start of every
   * job -- measured as the pool's cost on decode, where the ops are small enough
   * for that to dominate: at 44 threads against 22, `rms_norm` went 0.57x,
   * `rope` 0.37x and `silu_mul` 0.34x, all *slower* with twice the cores.
   *
   * The index is `thread_local`, written once when the worker starts, so a job's
   * per-worker cost is the barrier and nothing else.  The mapping is the same
   * partition `partition_size` describes: chunk *i* is
   * `[total*i/chunks, total*(i+1)/chunks)`.  The caller's thread is chunk zero
   * by never having set the variable. */

  /* A background worker's turn: take whatever chunks are left, then tell the
   * barrier it is out of the job.  The arrival is published *after* the last
   * chunk, with release ordering, so the caller returning from the barrier
   * cannot observe the arrival before the memory that chunk wrote.  The worker
   * then goes straight back to the spin loop rather than to a wait: the caller's
   * next job is usually tens of microseconds away, and sleeping and waking costs
   * more than the wait. */
  void run_worker_job() {
    const int64_t chunk = t_chunk_index;
    if (chunk < chunks_) {
      job_(chunk);
    }
    arrived_in_job_.fetch_add(1, std::memory_order_release);
  }

  void worker_main() {
    t_chunk_index = next_worker_index_.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t seen = 0;
    for (;;) {
      while (generation_.load(std::memory_order_acquire) == seen) {
        if (stop_.load(std::memory_order_relaxed)) {
          return;
        }
        POCKETLLM_PAUSE();
      }
      /* The generation may have moved because the pool is stopping rather than
       * because a job is pending -- the destructor bumps it to wake the
       * spinners.  Without this second check a worker would wake, read the
       * *previous* job's `chunks_` and `job_`, and call a body that captures a
       * frame which is being destroyed. */
      if (stop_.load(std::memory_order_relaxed)) {
        return;
      }
      seen = generation_;
      run_worker_job();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  /* Held only while a job's state is published and while the pool is stopping;
   * the workers never touch it, which is what lets the barrier be a spin. */
  std::function<void(int64_t)> job_;
  /* Only read by a worker to decide whether its index is inside this job; the
   * assignment itself needs no atomic -- see `run_worker_job`. */
  int64_t chunks_ = 0;
  /* Drains once, as the workers start, so `t_chunk_index` is stable for the
   * life of the pool. */
  std::atomic<int64_t> next_worker_index_{0};
  /* Job barrier: how many background workers this job started with, and how
   * many have finished it.  The caller contributes no arrival -- it runs
   * own chunk itself and then spins until the rest arrive. */
  int64_t workers_left_ = 0;
  std::atomic<int64_t> arrived_in_job_{0};
  std::atomic<uint64_t> generation_{0};
  std::atomic<bool> stop_{false};
};

}  // namespace detail

/* Run ``fn(lo, hi)`` over a contiguous partition of ``[0, total)``.
 *
 * ``min_per_task`` is the smallest span worth waking a thread for; pick it from
 * the cost of one iteration so a tiny op stays serial. */
template <typename Fn>
void parallel_for(int64_t total, int64_t min_per_task, Fn &&fn) {
  detail::ThreadPool::instance().run_ranges(total, min_per_task <= 0 ? 1 : min_per_task,
                                            static_cast<Fn &&>(fn));
}

/* The number of tasks :func:`parallel_for` will run for a job of this shape.
 *
 * A caller that gives every task a private scratch region -- the CPU backend's
 * ``attention_scratch``, which sizes the score rows -- has to arrive at the same
 * partition the kernel will take.  Both go through this one function, so they
 * cannot disagree. */
inline int64_t parallel_tasks(int64_t total, int64_t min_per_task) {
  if (min_per_task <= 0) {
    min_per_task = 1;
  }
  return detail::partition_size(total, min_per_task, detail::ThreadPool::instance().size());
}

}  // namespace kernel
}  // namespace pocketllm

#endif /* POCKETLLM_KERNEL_PARALLEL_H */