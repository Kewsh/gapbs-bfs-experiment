#!/usr/bin/env bash
# Run GAPBS PageRank and IMC CAS bandwidth sampler together.
# The sampler starts counting only after graph build (SIGUSR1 from pr).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/gapbs_config.env"

INTERVAL_US="${INTERVAL_US}"
OUTPUT_CSV=""
RUN_NAME=""
SKIP_PR=0
USE_PTHREAD=0
SAMPLER_PID=""
PR_PID=""
PR_LOG=""

usage() {
  cat <<EOF
Usage: sudo $0 [options]

Run GAPBS PageRank (Kronecker synthetic graph) and imc_bw_sampler concurrently.
IMC counting begins after graph build, when pr sends SIGUSR1 to the sampler.

Options:
  -i, --interval-us US   Sample interval (default: ${INTERVAL_US})
  -o, --output FILE      CSV output path
  -n, --name NAME        Label for output filenames
  -g, --scale N          Kronecker scale (2^N vertices, default: ${PR_SCALE})
  -k, --degree D         Average degree (default: ${PR_DEGREE})
  -t, --trials N         Timed PageRank trials (default: ${PR_TRIALS})
      --tolerance T      Convergence tolerance (default: ${PR_TOLERANCE})
      --max-iters N      Max PageRank iterations per trial (default: ${PR_MAX_ITERS})
  -j, --threads N        Thread count (OMP_NUM_THREADS / GAPBS_NUM_THREADS)
      --pthread            Use pr_pthread instead of OpenMP pr
      --turbo            Aggressive preset: shorter graph, 144 threads
      --sampler-only     Start sampler and wait (debug)
      --skip-pr          Only run sampler until Ctrl-C
  -h, --help             Show help

Environment:
  Edit ${ROOT}/gapbs_config.env for NUMA pinning and PR_EXTRA_ARGS.

Examples:
  sudo $0
  sudo $0 --interval-us 10 -g 24 -t 32 -n kr24
  sudo $0 -g 23 --max-iters 20 --tolerance 1e-4 -n sweet
  sudo $0 --turbo -n turbo_pr
EOF
}

TURBO_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--interval-us) INTERVAL_US="$2"; shift 2 ;;
    -o|--output) OUTPUT_CSV="$2"; shift 2 ;;
    -n|--name) RUN_NAME="$2"; shift 2 ;;
    -g|--scale) PR_SCALE="$2"; shift 2 ;;
    -k|--degree) PR_DEGREE="$2"; shift 2 ;;
    -t|--trials) PR_TRIALS="$2"; shift 2 ;;
    --tolerance) PR_TOLERANCE="$2"; shift 2 ;;
    --max-iters) PR_MAX_ITERS="$2"; shift 2 ;;
    -j|--threads) OMP_NUM_THREADS="$2"; shift 2 ;;
    --pthread) USE_PTHREAD=1; shift ;;
    --turbo)
      TURBO_MODE=1
      PR_SCALE="${TURBO_SCALE}"
      OMP_NUM_THREADS="${TURBO_THREADS}"
      NUMA_CMD="${TURBO_NUMA_CMD}"
      shift
      ;;
    --sampler-only|--skip-pr) SKIP_PR=1; shift ;;
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
  if [[ ! -x "${ROOT}/pr_pthread" ]]; then
    make -C "${ROOT}" -f Makefile.pthread pr_pthread
  fi
  PR_BIN="${ROOT}/pr_pthread"
else
  if [[ ! -x "${ROOT}/pr" ]]; then
    make -C "${ROOT}" pr
  fi
  PR_BIN="${ROOT}/pr"
fi

mkdir -p "${ROOT}/${RESULTS_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
suffix="${RUN_NAME:-pr_g${PR_SCALE}_k${PR_DEGREE}}"
if [[ -z "${OUTPUT_CSV}" ]]; then
  OUTPUT_CSV="${ROOT}/${RESULTS_DIR}/imc_bw_${INTERVAL_US}us_${suffix}_${timestamp}.csv"
fi
PR_LOG="${ROOT}/${RESULTS_DIR}/pr_${suffix}_${timestamp}.log"

export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES
export GAPBS_NUM_THREADS="${OMP_NUM_THREADS}"
if [[ "${USE_PTHREAD}" -eq 1 ]]; then
  export OMP_WAIT_POLICY=passive
  export KMP_BLOCKTIME=0
fi

build_pr_args() {
  PR_ARGS=(-g "${PR_SCALE}" -k "${PR_DEGREE}" -n "${PR_TRIALS}" \
    -t "${PR_TOLERANCE}" -i "${PR_MAX_ITERS}")
  if [[ -n "${PR_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=(${PR_EXTRA_ARGS})
    PR_ARGS+=("${EXTRA_ARR[@]}")
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

run_pr() {
  build_pr_args
  export GAPBS_SAMPLER_PID="${SAMPLER_PID}"

  echo "Starting PageRank (graph build + ${PR_TRIALS} trials) ..."
  echo "PR: tolerance=${PR_TOLERANCE} max_iters=${PR_MAX_ITERS}" >>"${PR_LOG}"
  echo "Command: ${NUMA_CMD} ${PR_BIN} ${PR_ARGS[*]}" | tee "${PR_LOG}"

  # shellcheck disable=SC2086
  ${NUMA_CMD} "${PR_BIN}" "${PR_ARGS[@]}" >>"${PR_LOG}" 2>&1 &
  PR_PID=$!
}

cleanup() {
  if [[ -n "${PR_PID}" ]] && kill -0 "${PR_PID}" 2>/dev/null; then
    kill -TERM "${PR_PID}" 2>/dev/null || true
    wait "${PR_PID}" 2>/dev/null || true
  fi
  unset GAPBS_SAMPLER_PID
  stop_sampler
}
trap cleanup EXIT INT TERM

echo "=== gapbs-pr-perf run ==="
echo "Interval:   ${INTERVAL_US} us"
echo "Graph:      Kronecker 2^${PR_SCALE} vertices, degree ${PR_DEGREE}"
echo "Trials:     ${PR_TRIALS}"
echo "Tolerance:  ${PR_TOLERANCE}"
echo "Max iters:  ${PR_MAX_ITERS}"
if [[ "${TURBO_MODE}" -eq 1 ]]; then
  echo "Preset:     turbo"
fi
echo "Threads:    ${OMP_NUM_THREADS}"
if [[ "${USE_PTHREAD}" -eq 1 ]]; then
  echo "Backend:    pthread (${PR_BIN})"
else
  echo "Backend:    OpenMP (${PR_BIN})"
fi
echo "NUMA:       ${NUMA_CMD}"
echo "CSV:        ${OUTPUT_CSV}"
echo "PR log:     ${PR_LOG}"
echo

if [[ "${SKIP_PR}" -eq 1 ]]; then
  start_sampler
  echo "Sampler PID ${SAMPLER_PID}. Press Ctrl-C to stop."
  wait "${SAMPLER_PID}"
  trap - EXIT INT TERM
  exit 0
fi

start_sampler
run_pr

set +e
wait "${PR_PID}"
PR_STATUS=$?
set -e

echo
echo "PageRank finished with status ${PR_STATUS}"
echo "Stopping sampler (PID ${SAMPLER_PID}) ..."
stop_sampler
trap - EXIT INT TERM

echo
echo "Done."
echo "  CSV: ${OUTPUT_CSV}"
echo "  Log: ${PR_LOG}"
echo
echo "Quick peek:"
head -n 5 "${OUTPUT_CSV}"
echo "..."
tail -n 3 "${OUTPUT_CSV}"

exit "${PR_STATUS}"
