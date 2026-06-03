#!/usr/bin/env bash
# Run GAPBS BFS and IMC CAS bandwidth sampler together.
# The sampler starts counting only after graph build (SIGUSR1 from bfs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/gapbs_config.env"

INTERVAL_US="${INTERVAL_US}"
OUTPUT_CSV=""
RUN_NAME=""
SKIP_BFS=0
SAMPLER_ONLY=0
USE_PTHREAD=0
SAMPLER_PID=""
BFS_PID=""
BFS_LOG=""

usage() {
  cat <<EOF
Usage: sudo $0 [options]

Run GAPBS BFS (Kronecker synthetic graph) and imc_bw_sampler concurrently.
IMC counting begins after graph build, when bfs sends SIGUSR1 to the sampler.

Options:
  -i, --interval-us US   Sample interval (default: ${INTERVAL_US})
  -o, --output FILE      CSV output path
  -n, --name NAME        Label for output filenames
  -g, --scale N          Kronecker scale (2^N vertices, default: ${BFS_SCALE})
  -k, --degree D         Average degree (default: ${BFS_DEGREE})
  -t, --trials N         Timed BFS trials (default: ${BFS_TRIALS})
  -j, --threads N        Thread count (OMP_NUM_THREADS / GAPBS_NUM_THREADS)
      --pthread            Use bfs_pthread instead of OpenMP bfs
      --alpha N          DOBFS alpha (default: ${BFS_ALPHA})
      --beta N           DOBFS beta (default: ${BFS_BETA})
      --max-bu N         Cap bottom-up iterations per epoch (0=off, 1=shortest BU spikes)
      --turbo            Aggressive preset: shorter graph, alpha=34, beta=2, max-bu=1, 144 threads
      --sampler-only     Start sampler and wait (debug)
      --skip-bfs         Only run sampler until Ctrl-C
  -h, --help             Show help

Environment:
  Edit ${ROOT}/gapbs_config.env for NUMA pinning and BFS_EXTRA_ARGS.

Examples:
  sudo $0
  sudo $0 --interval-us 10 -g 24 -t 32 -n kr24
  sudo $0 -g 23 --alpha 28 --beta 4 -n sweet
  sudo $0 --turbo -n turbo_max_spikes
EOF
}

TURBO_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--interval-us) INTERVAL_US="$2"; shift 2 ;;
    -o|--output) OUTPUT_CSV="$2"; shift 2 ;;
    -n|--name) RUN_NAME="$2"; shift 2 ;;
    -g|--scale) BFS_SCALE="$2"; shift 2 ;;
    -k|--degree) BFS_DEGREE="$2"; shift 2 ;;
    -t|--trials) BFS_TRIALS="$2"; shift 2 ;;
    -j|--threads) OMP_NUM_THREADS="$2"; shift 2 ;;
    --pthread) USE_PTHREAD=1; shift ;;
    --alpha) BFS_ALPHA="$2"; shift 2 ;;
    --beta) BFS_BETA="$2"; shift 2 ;;
    --max-bu) BFS_MAX_BU="$2"; shift 2 ;;
    --turbo)
      TURBO_MODE=1
      BFS_SCALE="${TURBO_SCALE}"
      BFS_ALPHA="${TURBO_ALPHA}"
      BFS_BETA="${TURBO_BETA}"
      BFS_MAX_BU="${TURBO_MAX_BU}"
      OMP_NUM_THREADS="${TURBO_THREADS}"
      NUMA_CMD="${TURBO_NUMA_CMD}"
      shift
      ;;
    --sampler-only|--skip-bfs) SKIP_BFS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script must run as root to access uncore IMC PMUs." >&2
  echo "Re-run with: sudo $0 $*" >&2
  exit 1
fi

if [[ ! -x "${ROOT}/imc_bw_sampler" ]]; then
  make -C "${ROOT}" imc_bw_sampler
fi
if [[ "${USE_PTHREAD}" -eq 1 ]]; then
  if [[ ! -x "${ROOT}/bfs_pthread" ]]; then
    make -C "${ROOT}" -f Makefile.pthread bfs_pthread
  fi
  BFS_BIN="${ROOT}/bfs_pthread"
else
  if [[ ! -x "${ROOT}/bfs" ]]; then
    make -C "${ROOT}" bfs
  fi
  BFS_BIN="${ROOT}/bfs"
fi

mkdir -p "${ROOT}/${RESULTS_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
suffix="${RUN_NAME:-bfs_g${BFS_SCALE}_k${BFS_DEGREE}}"
if [[ -z "${OUTPUT_CSV}" ]]; then
  OUTPUT_CSV="${ROOT}/${RESULTS_DIR}/imc_bw_${INTERVAL_US}us_${suffix}_${timestamp}.csv"
fi
BFS_LOG="${ROOT}/${RESULTS_DIR}/bfs_${suffix}_${timestamp}.log"

export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES
export GAPBS_NUM_THREADS="${OMP_NUM_THREADS}"
if [[ "${USE_PTHREAD}" -eq 1 ]]; then
  export OMP_WAIT_POLICY=passive
  export KMP_BLOCKTIME=0
fi

build_bfs_args() {
  BFS_ARGS=(-g "${BFS_SCALE}" -k "${BFS_DEGREE}" -n "${BFS_TRIALS}")
  if [[ -n "${BFS_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=(${BFS_EXTRA_ARGS})
    BFS_ARGS+=("${EXTRA_ARR[@]}")
  fi
}

start_sampler() {
  echo "Starting IMC sampler (wait for graph ready) -> ${OUTPUT_CSV}"
  "${ROOT}/imc_bw_sampler" --wait-sigusr1 --interval-us "${INTERVAL_US}" \
    --output "${OUTPUT_CSV}" &
  SAMPLER_PID=$!
  sleep 0.05
}

stop_sampler() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill -INT "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
}

run_bfs() {
  build_bfs_args
  export GAPBS_SAMPLER_PID="${SAMPLER_PID}"
  export GAPBS_BFS_ALPHA="${BFS_ALPHA}"
  export GAPBS_BFS_BETA="${BFS_BETA}"
  export GAPBS_BFS_MAX_BU="${BFS_MAX_BU}"

  echo "Starting BFS (graph build + ${BFS_TRIALS} trials) ..."
  echo "DOBFS: alpha=${BFS_ALPHA} beta=${BFS_BETA} max_bu=${BFS_MAX_BU}" >>"${BFS_LOG}"
  echo "Command: ${NUMA_CMD} ${BFS_BIN} ${BFS_ARGS[*]}" | tee "${BFS_LOG}"

  # shellcheck disable=SC2086
  ${NUMA_CMD} "${BFS_BIN}" "${BFS_ARGS[@]}" >>"${BFS_LOG}" 2>&1 &
  BFS_PID=$!
}

cleanup() {
  if [[ -n "${BFS_PID}" ]] && kill -0 "${BFS_PID}" 2>/dev/null; then
    kill -TERM "${BFS_PID}" 2>/dev/null || true
    wait "${BFS_PID}" 2>/dev/null || true
  fi
  unset GAPBS_SAMPLER_PID
  stop_sampler
}
trap cleanup EXIT INT TERM

echo "=== gapbs-perf run ==="
echo "Interval:   ${INTERVAL_US} us"
echo "Graph:      Kronecker 2^${BFS_SCALE} vertices, degree ${BFS_DEGREE}"
echo "Trials:     ${BFS_TRIALS}"
echo "DOBFS:      alpha=${BFS_ALPHA} beta=${BFS_BETA} max_bu=${BFS_MAX_BU}"
if [[ "${TURBO_MODE}" -eq 1 ]]; then
  echo "Preset:     turbo"
fi
echo "Threads:    ${OMP_NUM_THREADS}"
if [[ "${USE_PTHREAD}" -eq 1 ]]; then
  echo "Backend:    pthread (${BFS_BIN})"
else
  echo "Backend:    OpenMP (${BFS_BIN})"
fi
echo "NUMA:       ${NUMA_CMD}"
echo "CSV:        ${OUTPUT_CSV}"
echo "BFS log:    ${BFS_LOG}"
echo

if [[ "${SKIP_BFS}" -eq 1 ]]; then
  start_sampler
  echo "Sampler PID ${SAMPLER_PID}. Press Ctrl-C to stop."
  wait "${SAMPLER_PID}"
  trap - EXIT INT TERM
  exit 0
fi

start_sampler
run_bfs

set +e
wait "${BFS_PID}"
BFS_STATUS=$?
set -e

echo
echo "BFS finished with status ${BFS_STATUS}"
echo "Stopping sampler (PID ${SAMPLER_PID}) ..."
stop_sampler
trap - EXIT INT TERM

echo
echo "Done."
echo "  CSV: ${OUTPUT_CSV}"
echo "  Log: ${BFS_LOG}"
echo
echo "Quick peek:"
head -n 5 "${OUTPUT_CSV}"
echo "..."
tail -n 3 "${OUTPUT_CSV}"

exit "${BFS_STATUS}"
