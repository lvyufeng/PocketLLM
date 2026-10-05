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
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace pocketllm {
namespace kernel {

/* How many threads the CPU kernels should use.
 *
 * `$POCKETLLM_CPU_THREADS` wins when it is set and parses to a positive number,
 * which is the escape hatch for reproducing a single-threaded number
 * (``POCKETLLM_CPU_THREADS=1``) and for staying a good citizen on a shared
 * host.  Otherwise every hardware thread is used: the requirement this exists
 * to meet is speed without the caller having to configure anything, and because
 * the result is independent of the thread count there is no correctness reason
 * to be conservative.  Clamped to [1, 256] so a typo cannot start a thousand
 * threads. */
inline int64_t cpu_thread_count() {
  const char *from_env = std::getenv("POCKETLLM_CPU_THREADS");
  if (from_env != nullptr) {
    char *end = nullptr;
    const long parsed = std::strtol(from_env, &end, 10);
    if (end != from_env && parsed > 0) {
      return parsed > 256 ? 256 : static_cast<int64_t>(parsed);
    }
  }
  const unsigned hardware = std::thread::hardware_concurrency();
  return hardware == 0 ? 1 : static_cast<int64_t>(hardware);
}

namespace detail {

/* How many contiguous ranges a job of `total` units splits into, given a pool
 * of `threads` and a `min_per_task` floor.
 *
 * Shared between the pool that runs the ranges and `parallel_tasks`, which a
 * caller uses to size per-task scratch; the two must agree or a buffer is sized
 * for a partition that does not happen.  It takes the thread count as an
 * argument so it is callable before the pool exists. */
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
 * parallel call inside another.  The mutex is there for the worker bookkeeping,
 * not for concurrent callers, and `job_` is a reference into the caller's frame
 * that outlives the call because the caller blocks at a barrier until every
 * worker has *left* the job -- not merely until every chunk has run.  A worker
 * between its last chunk and its next claim is still reading the job, so the
 * caller cannot return while one is in that window or the next job's
 * `next_chunk_` reset would race it.
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

    {
      std::lock_guard<std::mutex> lock(mutex_);
      /* A reference into the caller's frame: valid because the caller blocks
       * below until every worker has left this job, and the caller's frame
       * cannot unwind while it is blocked here. */
      job_ = [&fn, total, chunks](int64_t chunk) {
        const int64_t lo = total * chunk / chunks;
        const int64_t hi = total * (chunk + 1) / chunks;
        fn(lo, hi, chunk);
      };
      chunks_ = chunks;
      next_chunk_.store(0, std::memory_order_relaxed);
      /* Not every worker necessarily takes a chunk on every job -- one can be
       * descheduled across the whole call -- so the barrier counts arrivals at
       * the *end* of the job rather than chunks claimed.  Every worker that is
       * running tells `end_job`; the caller knows the count. */
      workers_left_ = static_cast<int64_t>(workers_.size());
      arrived_in_job_ = 0;
      ++generation_;
    }
    cv_.notify_all();

    /* This thread is a worker too -- one core would otherwise sit idle for the
     * whole call, which is a fifth of the budget on a four-core phone. */
    work();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return arrived_in_job_ == workers_left_; });
  }

 private:
  ThreadPool() {
    const int64_t wanted = cpu_thread_count();
    workers_.reserve(static_cast<std::size_t>(wanted));
    /* ``wanted - 1`` background threads: the calling thread is the last
     * executor, so a pool of N uses N cores rather than N+1. */
    for (int64_t i = 1; i < wanted; ++i) {
      workers_.emplace_back([this] { worker_main(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (std::thread &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  /* Claim and run chunks until none are left.  Calling `fetch_add` past the
   * last chunk is how a worker learns there is nothing more to do; because the
   * claim is atomic, no chunk is run twice and none is skipped even if a worker
   * wakes late.  When the chunks run out the worker still has to reach the job
   * barrier, or the caller's frame would be torn down while it was reading the
   * claim counter. */
  void work() {
    for (;;) {
      const int64_t chunk = next_chunk_.fetch_add(1, std::memory_order_relaxed);
      if (chunk >= chunks_) {
        return;
      }
      job_(chunk);
    }
  }

  /* A background worker's turn: take whatever chunks are left, then tell the
   * barrier it is out of the job.  The decrement happens before the next
   * `cv_.wait`, so the worker cannot be asleep in `wait` while still counted as
   * running -- `run_ranges` would hang waiting for an arrival that will not
   * come until the *next* job wakes it. */
  void run_worker_job() {
    work();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++arrived_in_job_;
      done_cv_.notify_all();
    }
  }

  void worker_main() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, seen] { return stop_ || generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
      }
      run_worker_job();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::function<void(int64_t)> job_;
  std::atomic<int64_t> next_chunk_{0};
  int64_t chunks_ = 0;
  /* Job barrier: how many background workers this job started with, and how
   * many have finished it.  The caller contributes no arrival -- it runs
   * `work()` to completion itself and then waits for the rest. */
  int64_t workers_left_ = 0;
  int64_t arrived_in_job_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
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