// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details
//
// Pthread port of direction-optimizing BFS (see bfs.cc).
//
// This file is written to mirror, as closely as a hand port can, the code GCC
// emits for the OpenMP kernel in bfs.cc:
//
//   * Each "#pragma omp parallel [for]" region becomes a dedicated worker
//     function with the loop body written *inline* -- exactly like GCC's
//     outlined `*._omp_fn.N` functions. Crucially the body is NOT reached
//     through a per-iteration function pointer, so the compiler can inline the
//     neighbourhood walk + bitmap probes and keep many independent memory
//     accesses in flight (the high memory-level parallelism that saturates
//     DRAM bandwidth in the bottom-up step).
//   * Loop scheduling matches libgomp: BUStep uses dynamic/1024 via a shared
//     atomic counter (GOMP_loop_nonmonotonic_dynamic), every other region uses
//     the default static schedule with GOMP's exact even split.
//   * Reductions accumulate into per-thread partials that main sums.
//   * The persistent worker team is released/collected with a sense-reversing
//     barrier that spins before sleeping, mirroring libgomp's barrier (glibc's
//     pthread_barrier_wait goes to a futex sleep almost immediately, which adds
//     wake-up latency to every one of the many small regions a BFS runs).

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <sys/syscall.h>
#include <linux/futex.h>

#include <atomic>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"


using namespace std;


static int GetNumThreads() {
  if (const char *env = getenv("GAPBS_NUM_THREADS")) {
    int n = atoi(env);
    if (n > 0)
      return n;
  }
  if (const char *env = getenv("OMP_NUM_THREADS")) {
    int n = atoi(env);
    if (n > 0)
      return n;
  }
  return max(1, static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN)));
}


// ---------------------------------------------------------------------------
// Low-latency team synchronization (libgomp-style spin-then-sleep barrier).
// ---------------------------------------------------------------------------

static inline void CpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  sched_yield();
#endif
}

static inline void FutexWait(int *addr, int expected) {
  syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
}

static inline void FutexWakeAll(int *addr) {
  syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}

// Number of CPUs this process is actually allowed to run on (respects
// taskset / numactl pinning), used to decide whether we are oversubscribed.
static int AvailableCPUs() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    int n = CPU_COUNT(&set);
    if (n > 0)
      return n;
  }
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? static_cast<int>(n) : 1;
}

// Pick a spin count the way libgomp's env.c does:
//   GOMP_SPINCOUNT (or our GAPBS_SPINCOUNT) overrides everything; otherwise the
//   default is 300000, throttled to 100 when the team is oversubscribed.
// We intentionally do NOT consult OMP_WAIT_POLICY: run_gapbs_perf.sh exports
// OMP_WAIT_POLICY=passive only on the pthread path, which would force the team
// to sleep and diverge from the OpenMP run (which uses the default policy).
static long ComputeSpinCount(int num_threads) {
  if (const char *s = getenv("GAPBS_SPINCOUNT")) {
    long v = atol(s);
    if (v >= 0)
      return v;
  }
  if (const char *s = getenv("GOMP_SPINCOUNT")) {
    long v = atol(s);
    if (v >= 0)
      return v;
  }
  const long kDefaultSpin = 300000;
  const long kThrottledSpin = 100;
  int avail = AvailableCPUs();
  if (avail > 0 && num_threads > avail)
    return kThrottledSpin;
  return kDefaultSpin;
}


// Reusable centralized sense-reversing barrier for `participants` threads.
// Spins up to `spin` iterations, then blocks on a futex. Drop-in for the
// pthread_barrier_t the pool used before (same N+1 rendezvous semantics).
class SpinBarrier {
 public:
  void init(int participants, long spin) {
    total_ = participants;
    spin_ = spin;
    __atomic_store_n(&count_, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&phase_, 0, __ATOMIC_RELAXED);
  }

  void destroy() {}

  void wait() {
    int my_phase = __atomic_load_n(&phase_, __ATOMIC_RELAXED);
    // ACQ_REL so the last arriver's read of `count_` happens-after every other
    // thread's pre-barrier writes (RMW release sequence), giving full barrier
    // ordering once we publish the phase flip below.
    int arrived = __atomic_add_fetch(&count_, 1, __ATOMIC_ACQ_REL);
    if (arrived == total_) {
      __atomic_store_n(&count_, 0, __ATOMIC_RELAXED);
      __atomic_store_n(&phase_, my_phase ^ 1, __ATOMIC_RELEASE);
      FutexWakeAll(&phase_);
    } else {
      long spins = spin_;
      while (__atomic_load_n(&phase_, __ATOMIC_ACQUIRE) == my_phase) {
        if (spins > 0) {
          spins--;
          CpuRelax();
        } else {
          FutexWait(&phase_, my_phase);
        }
      }
    }
  }

 private:
  int total_ = 0;
  long spin_ = 0;
  int count_ = 0;   // arrivals this episode
  int phase_ = 0;   // sense / futex word
};


/*
 * Persistent worker pool: n workers + main, released and collected with two
 * barriers per region (mirrors libgomp's "wake team" + "end-of-parallel
 * barrier"). Same control flow as before; only the barrier primitive changed.
 */
class ThreadPool {
 public:
  ThreadPool()
      : num_threads_(0), shutdown_(false),
        task_fn_(nullptr), task_ctx_(nullptr) {}

  ~ThreadPool() {
    Shutdown();
  }

  void Init(int num_threads) {
    if (num_threads_ > 0)
      return;
    num_threads_ = num_threads;
    worker_self_ = this;
    long spin = ComputeSpinCount(num_threads_);
    job_start_.init(num_threads_ + 1, spin);
    job_end_.init(num_threads_ + 1, spin);
    thread_ids_.resize(num_threads_);
    threads_.resize(num_threads_);
    for (int t = 0; t < num_threads_; t++) {
      thread_ids_[t] = t;
      pthread_create(&threads_[t], nullptr, WorkerMain, &thread_ids_[t]);
    }
  }

  void Shutdown() {
    if (num_threads_ == 0)
      return;
    shutdown_ = true;
    job_start_.wait();
    job_end_.wait();
    for (int t = 0; t < num_threads_; t++)
      pthread_join(threads_[t], nullptr);
    job_start_.destroy();
    job_end_.destroy();
    num_threads_ = 0;
    threads_.clear();
    thread_ids_.clear();
    shutdown_ = false;
    worker_self_ = nullptr;
  }

  void Run(void (*fn)(int, int, void *), void *ctx) {
    task_fn_ = fn;
    task_ctx_ = ctx;
    job_start_.wait();
    job_end_.wait();
    task_fn_ = nullptr;
    task_ctx_ = nullptr;
  }

  int num_threads() const { return num_threads_; }

 private:
  static ThreadPool *worker_self_;

  static void *WorkerMain(void *arg) {
    int tid = *static_cast<int *>(arg);
    ThreadPool &pool = *worker_self_;
    while (true) {
      pool.job_start_.wait();
      if (pool.shutdown_)
        break;
      pool.task_fn_(tid, pool.num_threads_, pool.task_ctx_);
      pool.job_end_.wait();
    }
    pool.job_end_.wait();
    return nullptr;
  }

  int num_threads_;
  vector<pthread_t> threads_;
  vector<int> thread_ids_;
  SpinBarrier job_start_;
  SpinBarrier job_end_;
  bool shutdown_;
  void (*task_fn_)(int, int, void *);
  void *task_ctx_;
};

ThreadPool *ThreadPool::worker_self_ = nullptr;


static ThreadPool &Pool() {
  static ThreadPool pool;
  static bool initialized = false;
  if (!initialized) {
    pool.Init(GetNumThreads());
    initialized = true;
  }
  return pool;
}


// GOMP's default-static distribution of [0, n) across `nthreads`: the first
// (n % nthreads) threads get ceil(n/nthreads), the rest floor(n/nthreads).
static inline void StaticRange(int64_t n, int nthreads, int tid,
                               int64_t *begin, int64_t *end) {
  int64_t q = n / nthreads;
  int64_t r = n % nthreads;
  if (tid < r) {
    *begin = (q + 1) * tid;
    *end = *begin + q + 1;
  } else {
    *begin = q * tid + r;
    *end = *begin + q;
  }
}


// ---------------------------------------------------------------------------
// BUStep -- mirrors:
//   #pragma omp parallel for reduction(+:awake_count) schedule(dynamic, 1024)
// ---------------------------------------------------------------------------
struct BUStepCtx {
  const Graph *g;
  pvector<NodeID> *parent;
  Bitmap *front;
  Bitmap *next;
  atomic<int64_t> *next_index;
  int64_t num_nodes;
  int64_t *awake_parts;
};

static void BUStepWorker(int tid, int /*nthreads*/, void *arg) {
  BUStepCtx *c = static_cast<BUStepCtx *>(arg);
  const Graph &g = *c->g;
  pvector<NodeID> &parent = *c->parent;
  Bitmap &front = *c->front;
  Bitmap &next = *c->next;
  const int64_t kChunk = 1024;
  int64_t awake_count = 0;
  while (true) {
    int64_t start = c->next_index->fetch_add(kChunk, memory_order_relaxed);
    if (start >= c->num_nodes)
      break;
    int64_t stop = min(start + kChunk, c->num_nodes);
    for (NodeID u = start; u < stop; u++) {
      if (parent[u] < 0) {
        for (NodeID v : g.in_neigh(u)) {
          if (front.get_bit(v)) {
            parent[u] = v;
            awake_count++;
            next.set_bit(u);
            break;
          }
        }
      }
    }
  }
  c->awake_parts[tid] = awake_count;
}

int64_t BUStep(const Graph &g, pvector<NodeID> &parent, Bitmap &front,
               Bitmap &next) {
  next.reset();
  ThreadPool &pool = Pool();
  vector<int64_t> awake_parts(pool.num_threads(), 0);
  atomic<int64_t> next_index(0);
  BUStepCtx ctx = {&g, &parent, &front, &next, &next_index, g.num_nodes(),
                   awake_parts.data()};
  pool.Run(BUStepWorker, &ctx);
  int64_t awake_count = 0;
  for (int64_t v : awake_parts)
    awake_count += v;
  return awake_count;
}


// ---------------------------------------------------------------------------
// TDStep -- mirrors:
//   #pragma omp parallel { QueueBuffer lqueue;
//     #pragma omp for reduction(+:scout_count) nowait { ... } lqueue.flush(); }
// (default static schedule, nowait -> the pool's end barrier is the region's
// implicit barrier).
// ---------------------------------------------------------------------------
struct TDStepCtx {
  const Graph *g;
  pvector<NodeID> *parent;
  SlidingQueue<NodeID> *queue;
  int64_t *scout_parts;
};

static void TDStepWorker(int tid, int nthreads, void *arg) {
  TDStepCtx *c = static_cast<TDStepCtx *>(arg);
  const Graph &g = *c->g;
  pvector<NodeID> &parent = *c->parent;
  QueueBuffer<NodeID> lqueue(*c->queue);
  auto q_begin = c->queue->begin();
  int64_t len = c->queue->end() - q_begin;
  int64_t my_begin, my_end;
  StaticRange(len, nthreads, tid, &my_begin, &my_end);

  int64_t scout_count = 0;
  for (int64_t i = my_begin; i < my_end; i++) {
    NodeID u = *(q_begin + i);
    for (NodeID v : g.out_neigh(u)) {
      NodeID curr_val = parent[v];
      if (curr_val < 0) {
        if (compare_and_swap(parent[v], curr_val, u)) {
          lqueue.push_back(v);
          scout_count += -curr_val;
        }
      }
    }
  }
  lqueue.flush();
  c->scout_parts[tid] = scout_count;
}

int64_t TDStep(const Graph &g, pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue) {
  ThreadPool &pool = Pool();
  vector<int64_t> scout_parts(pool.num_threads(), 0);
  TDStepCtx ctx = {&g, &parent, &queue, scout_parts.data()};
  pool.Run(TDStepWorker, &ctx);
  int64_t scout_count = 0;
  for (int64_t v : scout_parts)
    scout_count += v;
  return scout_count;
}


// ---------------------------------------------------------------------------
// QueueToBitmap -- mirrors: #pragma omp parallel for (default static)
// ---------------------------------------------------------------------------
struct QueueToBitmapCtx {
  SlidingQueue<NodeID>::iterator begin;
  int64_t size;
  Bitmap *bm;
};

static void QueueToBitmapWorker(int tid, int nthreads, void *arg) {
  QueueToBitmapCtx *c = static_cast<QueueToBitmapCtx *>(arg);
  Bitmap &bm = *c->bm;
  int64_t my_begin, my_end;
  StaticRange(c->size, nthreads, tid, &my_begin, &my_end);
  for (int64_t i = my_begin; i < my_end; i++) {
    NodeID u = *(c->begin + i);
    bm.set_bit_atomic(u);
  }
}

void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  QueueToBitmapCtx ctx = {queue.begin(), static_cast<int64_t>(queue.size()),
                          &bm};
  Pool().Run(QueueToBitmapWorker, &ctx);
}


// ---------------------------------------------------------------------------
// BitmapToQueue -- mirrors:
//   #pragma omp parallel { QueueBuffer lqueue;
//     #pragma omp for nowait { ... } lqueue.flush(); }
// (default static schedule)
// ---------------------------------------------------------------------------
struct BitmapToQueueCtx {
  const Graph *g;
  Bitmap *bm;
  SlidingQueue<NodeID> *queue;
};

static void BitmapToQueueWorker(int tid, int nthreads, void *arg) {
  BitmapToQueueCtx *c = static_cast<BitmapToQueueCtx *>(arg);
  Bitmap &bm = *c->bm;
  QueueBuffer<NodeID> lqueue(*c->queue);
  int64_t my_begin, my_end;
  StaticRange(c->g->num_nodes(), nthreads, tid, &my_begin, &my_end);
  for (NodeID n = my_begin; n < my_end; n++)
    if (bm.get_bit(n))
      lqueue.push_back(n);
  lqueue.flush();
}

void BitmapToQueue(const Graph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
  BitmapToQueueCtx ctx = {&g, const_cast<Bitmap *>(&bm), &queue};
  Pool().Run(BitmapToQueueWorker, &ctx);
  queue.slide_window();
}


// ---------------------------------------------------------------------------
// InitParent -- mirrors: #pragma omp parallel for (default static)
// ---------------------------------------------------------------------------
struct InitParentCtx {
  const Graph *g;
  pvector<NodeID> *parent;
};

static void InitParentWorker(int tid, int nthreads, void *arg) {
  InitParentCtx *c = static_cast<InitParentCtx *>(arg);
  const Graph &g = *c->g;
  pvector<NodeID> &parent = *c->parent;
  int64_t my_begin, my_end;
  StaticRange(g.num_nodes(), nthreads, tid, &my_begin, &my_end);
  for (NodeID n = my_begin; n < my_end; n++)
    parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
}

pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
  InitParentCtx ctx = {&g, &parent};
  Pool().Run(InitParentWorker, &ctx);
  return parent;
}


// ---------------------------------------------------------------------------
// Final cleanup sweep -- mirrors the closing #pragma omp parallel for in DOBFS.
// ---------------------------------------------------------------------------
struct CleanupParentCtx {
  pvector<NodeID> *parent;
  int64_t num_nodes;
};

static void CleanupParentWorker(int tid, int nthreads, void *arg) {
  CleanupParentCtx *c = static_cast<CleanupParentCtx *>(arg);
  pvector<NodeID> &parent = *c->parent;
  int64_t my_begin, my_end;
  StaticRange(c->num_nodes, nthreads, tid, &my_begin, &my_end);
  for (NodeID n = my_begin; n < my_end; n++)
    if (parent[n] < -1)
      parent[n] = -1;
}


pvector<NodeID> DOBFS(const Graph &g, NodeID source, bool logging_enabled = false,
                      int alpha = 15, int beta = 18, int max_bu_iters = 0) {
  if (logging_enabled)
    PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  t.Start();
  pvector<NodeID> parent = InitParent(g);
  t.Stop();
  if (logging_enabled)
    PrintStep("i", t.Seconds());
  parent[source] = source;
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = g.out_degree(source);
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      TIME_OP(t, QueueToBitmap(queue, front));
      if (logging_enabled)
        PrintStep("e", t.Seconds());
      awake_count = queue.size();
      queue.slide_window();
      int bu_iters = 0;
      do {
        t.Start();
        old_awake_count = awake_count;
        awake_count = BUStep(g, parent, front, curr);
        front.swap(curr);
        t.Stop();
        if (logging_enabled)
          PrintStep("bu", t.Seconds(), awake_count);
        bu_iters++;
      } while ((max_bu_iters == 0 || bu_iters < max_bu_iters) &&
               ((awake_count >= old_awake_count) ||
                (awake_count > g.num_nodes() / beta)));
      TIME_OP(t, BitmapToQueue(g, front, queue));
      if (logging_enabled)
        PrintStep("c", t.Seconds());
      scout_count = 1;
    } else {
      t.Start();
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue);
      queue.slide_window();
      t.Stop();
      if (logging_enabled)
        PrintStep("td", t.Seconds(), queue.size());
    }
  }
  CleanupParentCtx cleanup_ctx = {&parent, g.num_nodes()};
  Pool().Run(CleanupParentWorker, &cleanup_ctx);
  return parent;
}


void PrintBFSStats(const Graph &g, const pvector<NodeID> &bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n] >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}


bool BFSVerifier(const Graph &g, NodeID source,
                 const pvector<NodeID> &parent) {
  pvector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u] != -1)) {
      if (u == source) {
        if (!((parent[u] == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u]) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u]) {
      cout << "Reachability mismatch" << endl;
      return false;
    }
  }
  return true;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "breadth-first search (pthread)");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();

  const char *sampler_pid = getenv("GAPBS_SAMPLER_PID");
  if (sampler_pid != nullptr && sampler_pid[0] != '\0') {
    pid_t pid = static_cast<pid_t>(strtol(sampler_pid, nullptr, 10));
    if (pid > 1) {
      kill(pid, SIGUSR1);
    }
  }

  SourcePicker<Graph> sp(g, cli.start_vertex());
  int alpha = 15;
  int beta = 18;
  int max_bu_iters = 0;
  if (const char *a = getenv("GAPBS_BFS_ALPHA")) {
    alpha = atoi(a);
  }
  if (const char *b_env = getenv("GAPBS_BFS_BETA")) {
    beta = atoi(b_env);
  }
  if (const char *m = getenv("GAPBS_BFS_MAX_BU")) {
    max_bu_iters = atoi(m);
  }

  auto BFSBound = [&sp, &cli, alpha, beta, max_bu_iters] (const Graph &g) {
    return DOBFS(g, sp.PickNext(), cli.logging_en(), alpha, beta, max_bu_iters);
  };
  SourcePicker<Graph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp] (const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}
