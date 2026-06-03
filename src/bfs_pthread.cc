// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details
//
// Pthread port of direction-optimizing BFS (see bfs.cc).

#include <atomic>
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


/*
 * Persistent worker pool synchronized with pthread barriers (n workers + main).
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
    thread_ids_.resize(num_threads_);
    threads_.resize(num_threads_);
    pthread_barrier_init(&job_start_, nullptr, num_threads_ + 1);
    pthread_barrier_init(&job_end_, nullptr, num_threads_ + 1);
    for (int t = 0; t < num_threads_; t++) {
      thread_ids_[t] = t;
      pthread_create(&threads_[t], nullptr, WorkerMain, &thread_ids_[t]);
    }
  }

  void Shutdown() {
    if (num_threads_ == 0)
      return;
    shutdown_ = true;
    pthread_barrier_wait(&job_start_);
    pthread_barrier_wait(&job_end_);
    for (int t = 0; t < num_threads_; t++)
      pthread_join(threads_[t], nullptr);
    pthread_barrier_destroy(&job_start_);
    pthread_barrier_destroy(&job_end_);
    num_threads_ = 0;
    threads_.clear();
    thread_ids_.clear();
    shutdown_ = false;
    worker_self_ = nullptr;
  }

  void Run(void (*fn)(int, int, void *), void *ctx) {
    task_fn_ = fn;
    task_ctx_ = ctx;
    pthread_barrier_wait(&job_start_);
    pthread_barrier_wait(&job_end_);
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
      pthread_barrier_wait(&pool.job_start_);
      if (pool.shutdown_)
        break;
      pool.task_fn_(tid, pool.num_threads_, pool.task_ctx_);
      pthread_barrier_wait(&pool.job_end_);
    }
    pthread_barrier_wait(&pool.job_end_);
    return nullptr;
  }

  int num_threads_;
  vector<pthread_t> threads_;
  vector<int> thread_ids_;
  pthread_barrier_t job_start_;
  pthread_barrier_t job_end_;
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


static void ParallelFor(int64_t begin, int64_t end,
                        void (*body)(int64_t, void *), void *ctx) {
  struct ParForCtx {
    int64_t begin;
    int64_t end;
    void (*body)(int64_t, void *);
    void *ctx;
  } par_ctx = {begin, end, body, ctx};

  auto worker = [](int tid, int nthreads, void *arg) {
    ParForCtx *p = static_cast<ParForCtx *>(arg);
    int64_t count = p->end - p->begin;
    int64_t chunk = (count + nthreads - 1) / nthreads;
    int64_t my_begin = p->begin + static_cast<int64_t>(tid) * chunk;
    int64_t my_end = min(my_begin + chunk, p->end);
    for (int64_t i = my_begin; i < my_end; i++)
      p->body(i, p->ctx);
  };

  Pool().Run(worker, &par_ctx);
}


struct ReduceCtx {
  int64_t begin;
  int64_t end;
  int chunk;
  atomic<int64_t> next;
  int64_t *partials;
  void (*body)(int64_t, int64_t *, void *);
  void *body_ctx;
};


static void ParallelForReduceDynamic(int64_t begin, int64_t end, int chunk,
                                     void (*body)(int64_t, int64_t *, void *),
                                     void *body_ctx, int64_t *out) {
  ThreadPool &pool = Pool();
  vector<int64_t> partials(pool.num_threads(), 0);
  ReduceCtx rctx;
  rctx.begin = begin;
  rctx.end = end;
  rctx.chunk = chunk;
  rctx.next.store(begin);
  rctx.partials = partials.data();
  rctx.body = body;
  rctx.body_ctx = body_ctx;

  auto worker = [](int tid, int nthreads, void *arg) {
    ReduceCtx *p = static_cast<ReduceCtx *>(arg);
    int64_t local = 0;
    while (true) {
      int64_t i = p->next.fetch_add(p->chunk);
      if (i >= p->end)
        break;
      int64_t limit = min(i + p->chunk, p->end);
      for (int64_t j = i; j < limit; j++)
        p->body(j, &local, p->body_ctx);
    }
    p->partials[tid] = local;
  };

  pool.Run(worker, &rctx);

  int64_t sum = 0;
  for (int64_t v : partials)
    sum += v;
  *out = sum;
}


struct BUStepCtx {
  const Graph *g;
  pvector<NodeID> *parent;
  Bitmap *front;
  Bitmap *next;
};


static void BUStepBody(int64_t u, int64_t *local, void *arg) {
  BUStepCtx *ctx = static_cast<BUStepCtx *>(arg);
  if ((*ctx->parent)[u] < 0) {
    for (NodeID v : ctx->g->in_neigh(u)) {
      if (ctx->front->get_bit(v)) {
        (*ctx->parent)[u] = v;
        (*local)++;
        ctx->next->set_bit(u);
        break;
      }
    }
  }
}


int64_t BUStep(const Graph &g, pvector<NodeID> &parent, Bitmap &front,
               Bitmap &next) {
  next.reset();
  BUStepCtx ctx = {&g, &parent, &front, &next};
  int64_t awake_count = 0;
  ParallelForReduceDynamic(0, g.num_nodes(), 1024, BUStepBody, &ctx,
                           &awake_count);
  return awake_count;
}


struct TDStepCtx {
  const Graph *g;
  pvector<NodeID> *parent;
  SlidingQueue<NodeID> *queue;
  int64_t *scout_count;
};


static void TDStepWorker(int tid, int nthreads, void *arg) {
  TDStepCtx *ctx = static_cast<TDStepCtx *>(arg);
  QueueBuffer<NodeID> lqueue(*ctx->queue);
  auto q_begin = ctx->queue->begin();
  auto q_end = ctx->queue->end();
  int64_t len = q_end - q_begin;
  int64_t chunk = (len + nthreads - 1) / nthreads;
  auto my_begin = q_begin + static_cast<int64_t>(tid) * chunk;
  auto my_end = min(q_begin + static_cast<int64_t>(tid + 1) * chunk, q_end);

  int64_t local_scout = 0;
  for (auto q_iter = my_begin; q_iter < my_end; q_iter++) {
    NodeID u = *q_iter;
    for (NodeID v : ctx->g->out_neigh(u)) {
      NodeID curr_val = (*ctx->parent)[v];
      if (curr_val < 0) {
        if (compare_and_swap((*ctx->parent)[v], curr_val, u)) {
          lqueue.push_back(v);
          local_scout += -curr_val;
        }
      }
    }
  }
  lqueue.flush();
  ctx->scout_count[tid] = local_scout;
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


struct QueueIterCtx {
  SlidingQueue<NodeID>::iterator begin;
  SlidingQueue<NodeID>::iterator end;
  Bitmap *bm;
};


static void QueueToBitmapBody(int64_t offset, void *arg) {
  QueueIterCtx *ctx = static_cast<QueueIterCtx *>(arg);
  NodeID u = *(ctx->begin + offset);
  ctx->bm->set_bit_atomic(u);
}


void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  QueueIterCtx ctx = {queue.begin(), queue.end(), &bm};
  ParallelFor(0, queue.size(), QueueToBitmapBody, &ctx);
}


struct BitmapToQueueCtx {
  const Graph *g;
  Bitmap *bm;
  SlidingQueue<NodeID> *queue;
};


static void BitmapToQueueWorker(int tid, int nthreads, void *arg) {
  BitmapToQueueCtx *ctx = static_cast<BitmapToQueueCtx *>(arg);
  QueueBuffer<NodeID> lqueue(*ctx->queue);
  NodeID n_begin = (ctx->g->num_nodes() * tid) / nthreads;
  NodeID n_end = (ctx->g->num_nodes() * (tid + 1)) / nthreads;
  for (NodeID n = n_begin; n < n_end; n++)
    if (ctx->bm->get_bit(n))
      lqueue.push_back(n);
  lqueue.flush();
}


void BitmapToQueue(const Graph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
  BitmapToQueueCtx ctx = {&g, const_cast<Bitmap *>(&bm), &queue};
  Pool().Run(BitmapToQueueWorker, &ctx);
  queue.slide_window();
}


struct InitParentCtx {
  const Graph *g;
  pvector<NodeID> *parent;
};


static void InitParentBody(int64_t n, void *arg) {
  InitParentCtx *ctx = static_cast<InitParentCtx *>(arg);
  (*ctx->parent)[n] = ctx->g->out_degree(n) != 0 ? -ctx->g->out_degree(n) : -1;
}


pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
  InitParentCtx ctx = {&g, &parent};
  ParallelFor(0, g.num_nodes(), InitParentBody, &ctx);
  return parent;
}


struct CleanupParentCtx {
  pvector<NodeID> *parent;
};


static void CleanupParentBody(int64_t n, void *arg) {
  CleanupParentCtx *ctx = static_cast<CleanupParentCtx *>(arg);
  if ((*ctx->parent)[n] < -1)
    (*ctx->parent)[n] = -1;
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
  CleanupParentCtx cleanup_ctx = {&parent};
  ParallelFor(0, g.num_nodes(), CleanupParentBody, &cleanup_ctx);
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
