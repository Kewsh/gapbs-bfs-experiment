// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details
//
// Caladan port of direction-optimizing BFS (see GAPBS bfs.cc / bfs_pthread.cc).
//
// This version runs on top of Caladan's runtime: every parallel region is a
// fork-join over Caladan green threads coordinated by an rt::WaitGroup, rather
// than a persistent pthread team.  This is the idiomatic Caladan structure --
// at each region we spawn the workers and block on the waitgroup, so between
// regions the runtime is free to park threads and let the IOKernel reallocate
// cores at microsecond granularity (the whole point of Caladan).  We therefore
// deliberately do NOT carry over the pthread version's persistent pool or its
// spinning futex barrier: spinning would defeat core reallocation, and raw
// futex/pthread primitives must not be used from inside Caladan threads.
//
// What is preserved verbatim from the GAPBS algorithm:
//   * the per-region worker bodies (neighbourhood walks, bitmap probes),
//   * lock-free updates via compare_and_swap / Bitmap::set_bit_atomic (these
//     are plain CPU atomics and are safe inside Caladan threads),
//   * BUStep's dynamic/1024 schedule via a shared atomic cursor, and the
//     static even split (GOMP-style) used by the other regions,
//   * reductions accumulated into per-thread partials summed by the caller.
//
// The worker signature (tid, nthreads, void *ctx) is kept so the bodies match
// the pthread port one-to-one; only the dispatch layer changed.

#include <atomic>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <signal.h>
#include <unistd.h>

// Caladan C++ runtime bindings (bindings/cc, added to the include path by the
// Makefile).  runtime.h must come first; it pulls in the C runtime headers.
#include "runtime.h"
#include "sync.h"
#include "thread.h"

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


// Number of worker threads per parallel region.  Resolved once, after the
// runtime is up (so RuntimeMaxCores() is valid), and reused everywhere so the
// reduction-partial arrays are always sized to match the dispatched team.
static int g_num_threads = 1;

static int ResolveNumThreads() {
  int n = static_cast<int>(rt::RuntimeMaxCores());
  if (n < 1)
    n = 1;
  if (const char *env = getenv("GAPBS_NUM_THREADS")) {
    int e = atoi(env);
    if (e > 0)
      n = e;
  }
  return n;
}


// ---------------------------------------------------------------------------
// Fork-join dispatch.  Equivalent to the pthread pool's Run(fn, ctx): spawn
// `nthreads` Caladan threads each running fn(tid, nthreads, ctx), then block
// until all have finished.  Because the caller blocks on wg.Wait(), any
// stack-allocated context/partials in the caller outlive every worker, so
// capturing them by pointer is safe.
//
// We run worker 0 inline on the calling thread and spawn the remaining
// nthreads-1, mirroring how libgomp's master thread participates in the team.
// This also means a single-core configuration spawns nothing.
// ---------------------------------------------------------------------------
static void RunParallel(int nthreads, void (*fn)(int, int, void *),
                        void *ctx) {
  if (nthreads <= 1) {
    fn(0, 1, ctx);
    return;
  }
  rt::WaitGroup wg(nthreads - 1);
  for (int t = 1; t < nthreads; t++) {
    rt::Spawn([fn, t, nthreads, ctx, &wg] {
      fn(t, nthreads, ctx);
      wg.Done();
    });
  }
  fn(0, nthreads, ctx);
  wg.Wait();
}


// GOMP's exact static even split: the first n%p workers get the ceiling.
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
// BUStep -- bottom-up sweep.
//   #pragma omp parallel for reduction(+:awake_count) schedule(dynamic, 1024)
// Dynamic/1024 is reproduced with a shared atomic cursor.
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
  int n = g_num_threads;
  vector<int64_t> awake_parts(n, 0);
  atomic<int64_t> next_index(0);
  BUStepCtx ctx = {&g, &parent, &front, &next, &next_index, g.num_nodes(),
                   awake_parts.data()};
  RunParallel(n, BUStepWorker, &ctx);
  int64_t awake_count = 0;
  for (int64_t v : awake_parts)
    awake_count += v;
  return awake_count;
}


// ---------------------------------------------------------------------------
// TDStep -- top-down sweep.
//   #pragma omp parallel { QueueBuffer lqueue;
//     #pragma omp for reduction(+:scout_count) nowait { ... } lqueue.flush(); }
// (default static schedule).
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
  int n = g_num_threads;
  vector<int64_t> scout_parts(n, 0);
  TDStepCtx ctx = {&g, &parent, &queue, scout_parts.data()};
  RunParallel(n, TDStepWorker, &ctx);
  int64_t scout_count = 0;
  for (int64_t v : scout_parts)
    scout_count += v;
  return scout_count;
}


// ---------------------------------------------------------------------------
// QueueToBitmap -- #pragma omp parallel for (default static)
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
  RunParallel(g_num_threads, QueueToBitmapWorker, &ctx);
}


// ---------------------------------------------------------------------------
// BitmapToQueue -- #pragma omp parallel { QueueBuffer; for nowait; flush; }
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
  RunParallel(g_num_threads, BitmapToQueueWorker, &ctx);
  queue.slide_window();
}


// ---------------------------------------------------------------------------
// InitParent -- #pragma omp parallel for (default static)
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
  RunParallel(g_num_threads, InitParentWorker, &ctx);
  return parent;
}


// ---------------------------------------------------------------------------
// Final cleanup sweep -- the closing #pragma omp parallel for in DOBFS.
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
  RunParallel(g_num_threads, CleanupParentWorker, &cleanup_ctx);
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


// The body of the original main(), run as the first Caladan thread.  All graph
// construction and benchmarking happens here because the parallel regions spawn
// Caladan threads, which only exist once the runtime is up.
static void BfsMain(int argc, char *argv[]) {
  CLApp cli(argc, argv, "breadth-first search (caladan)");
  if (!cli.ParseArgs())
    return;

  // Worker count is fixed by the runtime's core budget (overridable by env).
  g_num_threads = ResolveNumThreads();

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
  if (const char *a = getenv("GAPBS_BFS_ALPHA"))
    alpha = atoi(a);
  if (const char *b_env = getenv("GAPBS_BFS_BETA"))
    beta = atoi(b_env);
  if (const char *m = getenv("GAPBS_BFS_MAX_BU"))
    max_bu_iters = atoi(m);

  auto BFSBound = [&sp, &cli, alpha, beta, max_bu_iters](const Graph &g) {
    return DOBFS(g, sp.PickNext(), cli.logging_en(), alpha, beta, max_bu_iters);
  };
  SourcePicker<Graph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp](const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
}


// Caladan entry point.  argv[1] is the runtime config file; the remaining
// arguments are the usual GAPBS command-line flags (-g/-n/-r/-v/...).
int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "usage: " << argv[0]
         << " <caladan_config> [gapbs args: -g <scale> -n <trials> "
            "-r <source> -v ...]"
         << endl;
    return -EINVAL;
  }

  string cfg = argv[1];

  // Re-pack argv for the GAPBS CLApp parser: program name, then args after the
  // config path.  The pointers reference argv, which lives for the whole
  // process, so capturing the vector by value is safe.
  vector<char *> app_argv;
  app_argv.push_back(argv[0]);
  for (int i = 2; i < argc; i++)
    app_argv.push_back(argv[i]);
  int app_argc = static_cast<int>(app_argv.size());

  int ret = rt::RuntimeInit(cfg, [app_argc, app_argv]() mutable {
    BfsMain(app_argc, app_argv.data());
  });
  if (ret) {
    cerr << "failed to start Caladan runtime" << endl;
    return ret;
  }
  return 0;
}
