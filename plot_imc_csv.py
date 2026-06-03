#!/usr/bin/env python3
"""Plot IMC CAS bandwidth samples from imc_bw_sampler CSV output."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def find_latest_csv(results_dir: Path) -> Path:
    matches = sorted(results_dir.glob("imc_bw_*.csv"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No imc_bw_*.csv files found in {results_dir}")
    return matches[-1]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "elapsed_us",
        "read_mib_per_s",
        "write_mib_per_s",
        "total_mib_per_s",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    # Absolute timestamps from the sampler (includes any pre-SIGUSR1 wait).
    df["elapsed_ms"] = df["elapsed_us"] / 1000.0
    df["elapsed_s"] = df["elapsed_us"] / 1_000_000.0
    add_relative_time_columns(df)
    return df


def add_relative_time_columns(df: pd.DataFrame) -> None:
    """Zero-base elapsed time at the first CSV row (start of measured kernel window)."""
    t0_us = float(df["elapsed_us"].iloc[0])
    df["elapsed_us_rel"] = df["elapsed_us"] - t0_us
    df["elapsed_ms_rel"] = df["elapsed_us_rel"] / 1000.0
    df["elapsed_s_rel"] = df["elapsed_us_rel"] / 1_000_000.0


def time_span_seconds(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    return (float(df["elapsed_us"].iloc[-1]) - float(df["elapsed_us"].iloc[0])) / 1_000_000.0


def select_plot_time_column(df: pd.DataFrame, time_unit: str, *, zero_based: bool) -> str:
    if zero_based:
        return "elapsed_ms_rel" if time_unit == "ms" else "elapsed_s_rel"
    return "elapsed_ms" if time_unit == "ms" else "elapsed_s"


def maybe_smooth(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def filter_time_window(
    df: pd.DataFrame,
    *,
    t_start: float | None,
    t_end: float | None,
    time_unit: str,
) -> pd.DataFrame:
    if t_start is None and t_end is None:
        return df

    time_col = "elapsed_ms" if time_unit == "ms" else "elapsed_s"
    filtered = df
    if t_start is not None:
        filtered = filtered[filtered[time_col] >= t_start]
    if t_end is not None:
        filtered = filtered[filtered[time_col] <= t_end]
    return filtered.reset_index(drop=True)


def apply_window_relative_time(
    df: pd.DataFrame,
    *,
    t_start: float,
    time_unit: str,
) -> pd.DataFrame:
    """Re-base x-axis to zero at the window start (requires --t-start/--t-end)."""
    out = df.copy()
    if time_unit == "ms":
        out["plot_ms"] = out["elapsed_ms"] - t_start
        out["plot_s"] = out["plot_ms"] / 1000.0
    else:
        out["plot_s"] = out["elapsed_s"] - t_start
        out["plot_ms"] = out["plot_s"] * 1000.0
    return out


def gbps_to_mibs(gbps: float) -> float:
    """Convert decimal GB/s (10^9 bytes/s) to MiB/s (matches CSV scale)."""
    return gbps * 1e9 / (1024**2)


def detect_memory_bursts(
    df: pd.DataFrame,
    *,
    threshold_mibs: float,
    min_duration_us: float,
    min_separation_us: float,
    bw_column: str = "total_mib_per_s",
) -> list[tuple[float, float]]:
    """
    Find high-BW bursts in the trace.

    A burst is a contiguous run of samples >= threshold lasting at least
    min_duration_us. Bursts whose starts are within min_separation_us of the
    previous burst's end are merged into one.
    """
    if df.empty:
        return []

    elapsed = df["elapsed_us"].to_numpy()
    bw = df[bw_column].to_numpy()
    interval = (
        df["interval_us"].to_numpy()
        if "interval_us" in df.columns
        else None
    )

    above = bw >= threshold_mibs
    if not above.any():
        return []

    raw: list[tuple[float, float]] = []
    i = 0
    n = len(above)
    while i < n:
        if not above[i]:
            i += 1
            continue
        start_idx = i
        while i < n and above[i]:
            i += 1
        end_idx = i - 1

        t_start = float(elapsed[start_idx])
        t_end = float(elapsed[end_idx])
        if interval is not None:
            t_end += float(interval[end_idx])
        duration_us = t_end - t_start

        if duration_us >= min_duration_us:
            raw.append((t_start, t_end))

    if not raw:
        return []

    merged: list[tuple[float, float]] = [raw[0]]
    for t_start, t_end in raw[1:]:
        gap_us = t_start - merged[-1][1]
        if gap_us < min_separation_us:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t_end))
        else:
            merged.append((t_start, t_end))
    return merged


@dataclass(frozen=True)
class BurstStats:
    count: int
    bursts_per_second: float
    analysis_duration_s: float
    threshold_gbps: float
    threshold_mibs: float
    min_duration_us: float
    min_separation_us: float

    def summary(self) -> str:
        return (
            f"bursts={self.count}  "
            f"bursts/s={self.bursts_per_second:.2f}  "
            f"(>{self.threshold_gbps:g} GB/s for >={self.min_duration_us:g} us, "
            f"merged if <{self.min_separation_us:g} us apart)"
        )


def compute_burst_stats(
    df: pd.DataFrame,
    bursts: list[tuple[float, float]],
    *,
    threshold_gbps: float,
    threshold_mibs: float,
    min_duration_us: float,
    min_separation_us: float,
) -> BurstStats:
    if len(df) < 2:
        duration_s = 0.0
    else:
        duration_s = (float(df["elapsed_us"].iloc[-1]) - float(df["elapsed_us"].iloc[0])) / 1e6
        if duration_s <= 0:
            duration_s = 1e-9

    count = len(bursts)
    bps = count / duration_s if duration_s > 0 else 0.0
    return BurstStats(
        count=count,
        bursts_per_second=bps,
        analysis_duration_s=duration_s,
        threshold_gbps=threshold_gbps,
        threshold_mibs=threshold_mibs,
        min_duration_us=min_duration_us,
        min_separation_us=min_separation_us,
    )


def burst_us_to_plot_x(t_us: float, df: pd.DataFrame, time_col: str) -> float:
    """Map absolute elapsed_us to the plot x-axis (no extrapolation outside samples)."""
    t_rel_us = t_us - float(df["elapsed_us"].iloc[0])
    xp = df["elapsed_us_rel"].to_numpy()
    fp = df[time_col].to_numpy()
    return float(np.interp(t_rel_us, xp, fp, left=fp[0], right=fp[-1]))


_TIME_LABELS = {
    "plot_ms": "Time from window start (ms)",
    "plot_s": "Time from window start (s)",
    "elapsed_ms_rel": "Time from first sample (ms)",
    "elapsed_s_rel": "Time from first sample (s)",
    "elapsed_ms": "Sampler elapsed (ms)",
    "elapsed_s": "Sampler elapsed (s)",
}


def parse_window(value: str) -> tuple[float, float]:
    for sep in (":", ",", "-"):
        if sep in value:
            parts = value.split(sep, 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                return float(parts[0]), float(parts[1])
    raise ValueError("window must be START:END, START,END, or START-END")


def plot_csv(
    df: pd.DataFrame,
    *,
    title: str,
    output: Path | None,
    show: bool,
    smooth: int,
    time_unit: str,
    relative_time: bool,
    window_label: str | None,
    burst_stats: BurstStats | None = None,
    burst_intervals: list[tuple[float, float]] | None = None,
    plot_time_col: str = "elapsed_ms_rel",
    duration_s: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if plot_time_col not in df.columns:
        raise ValueError(f"plot time column missing: {plot_time_col}")
    time_col = plot_time_col
    time_label = _TIME_LABELS.get(time_col, "Time")

    read = maybe_smooth(df["read_mib_per_s"], smooth)
    write = maybe_smooth(df["write_mib_per_s"], smooth)
    total = maybe_smooth(df["total_mib_per_s"], smooth)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])

    ax_bw = axes[0]
    ax_bw.plot(df[time_col], read, label="Read", color="#1f77b4", linewidth=0.9, alpha=0.85)
    ax_bw.plot(df[time_col], write, label="Write", color="#ff7f0e", linewidth=0.9, alpha=0.85)
    ax_bw.plot(df[time_col], total, label="Total", color="#2ca02c", linewidth=1.2)
    ax_bw.set_ylabel("IMC CAS bandwidth (MiB/s)")
    ax_bw.set_title(title if not window_label else f"{title}\n{window_label}")
    ax_bw.grid(True, alpha=0.25)
    ax_bw.legend(loc="upper right")

    if burst_intervals and burst_stats is not None:
        threshold_mibs = burst_stats.threshold_mibs
        for t_start_us, t_end_us in burst_intervals:
            x0 = burst_us_to_plot_x(t_start_us, df, time_col)
            x1 = burst_us_to_plot_x(t_end_us, df, time_col)
            ax_bw.axvspan(x0, x1, color="#d62728", alpha=0.12, linewidth=0)
        ax_bw.axhline(
            threshold_mibs,
            color="#d62728",
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
            label=f"burst threshold ({burst_stats.threshold_gbps:g} GB/s)",
        )
        ax_bw.legend(loc="upper right")

    ax_cas = axes[1]
    if "cas_read_delta" in df.columns and "cas_write_delta" in df.columns:
        ax_cas.plot(
            df[time_col],
            maybe_smooth(df["cas_read_delta"], smooth),
            label="CAS read Δ",
            color="#1f77b4",
            linewidth=0.8,
            alpha=0.75,
        )
        ax_cas.plot(
            df[time_col],
            maybe_smooth(df["cas_write_delta"], smooth),
            label="CAS write Δ",
            color="#ff7f0e",
            linewidth=0.8,
            alpha=0.75,
        )
        ax_cas.plot(
            df[time_col],
            maybe_smooth(df["cas_total_delta"], smooth),
            label="CAS total Δ",
            color="#2ca02c",
            linewidth=1.0,
        )
        ax_cas.set_ylabel("CAS count Δ / sample")
    else:
        ax_cas.text(
            0.5,
            0.5,
            "CAS delta columns not found",
            ha="center",
            va="center",
            transform=ax_cas.transAxes,
        )

    ax_cas.set_xlabel(time_label)
    ax_cas.grid(True, alpha=0.25)
    ax_cas.legend(loc="upper right")

    if duration_s is None:
        duration_s = time_span_seconds(df)
    peak_total = df["total_mib_per_s"].max()
    mean_total = df["total_mib_per_s"].mean()
    x_end = float(df[time_col].iloc[-1] - df[time_col].iloc[0])
    x_unit = "ms" if "ms" in time_col else "s"
    footer = (
        f"samples={len(df):,}  duration={duration_s:.3f} s  "
        f"x-axis=0–{x_end:.2f} {x_unit}  "
        f"peak_total={peak_total:.1f} MiB/s  mean_total={mean_total:.1f} MiB/s"
    )
    if burst_stats is not None:
        footer += f"  |  {burst_stats.summary()}"
    fig.text(
        0.01,
        0.01,
        footer,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.35",
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Wrote {output}")

    if show:
        plt.show()
    elif output is None:
        # Non-interactive default: save next to CSV if no output specified.
        raise ValueError("Specify --output or --show")

    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot IMC CAS bandwidth CSV produced by imc_bw_sampler."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        help="Input CSV path. Defaults to the newest imc_bw_*.csv in results/.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image path (default: <csv_basename>.png beside the CSV)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling average window in samples (default: 1 = no smoothing)",
    )
    parser.add_argument(
        "--time-unit",
        choices=("ms", "s"),
        default="ms",
        help="X-axis time unit (default: ms)",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=None,
        metavar="T",
        help="Plot window start time (in --time-unit)",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        metavar="T",
        help="Plot window end time (in --time-unit)",
    )
    parser.add_argument(
        "--window",
        default=None,
        metavar="START:END",
        help="Shorthand for --t-start/--t-end, e.g. 35000:36000",
    )
    parser.add_argument(
        "--relative-time",
        action="store_true",
        default=False,
        help="Start x-axis at 0 relative to --t-start (default when a window is set)",
    )
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        default=False,
        help="Use sampler-absolute elapsed time on x-axis (non-zero start after SIGUSR1 wait)",
    )
    parser.add_argument(
        "--no-zero-time",
        action="store_false",
        dest="zero_time",
        help="Use sampler-absolute timestamps on x-axis (default: zero at first sample)",
    )
    parser.set_defaults(zero_time=True)
    parser.add_argument(
        "--results-dir",
        default=str(root / "results"),
        help="Directory searched for the latest CSV when input is omitted",
    )
    peak = parser.add_argument_group("burst / peak detection")
    peak.add_argument(
        "--peak-threshold-gbps",
        type=float,
        default=100.0,
        metavar="GB/s",
        help="Count bursts when total bandwidth exceeds this (decimal GB/s, default: 100)",
    )
    peak.add_argument(
        "--peak-min-duration-us",
        type=float,
        default=500.0,
        metavar="US",
        help="Burst must stay above threshold for at least this long (default: 500)",
    )
    peak.add_argument(
        "--peak-min-separation-us",
        type=float,
        default=500.0,
        metavar="US",
        help="Peaks closer than this are merged into one (default: 500)",
    )
    peak.add_argument(
        "--no-peak-stats",
        action="store_true",
        help="Skip burst detection and bursts/s reporting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
    else:
        csv_path = find_latest_csv(Path(args.results_dir).expanduser().resolve())

    if not csv_path.is_file():
        parser.error(f"CSV not found: {csv_path}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    elif args.show:
        output_path = None
    else:
        output_path = csv_path.with_suffix(".png")

    df = load_csv(csv_path)
    if df.empty:
        parser.error(f"CSV is empty: {csv_path}")

    t_start = args.t_start
    t_end = args.t_end
    if args.window is not None:
        if t_start is not None or t_end is not None:
            parser.error("Use either --window or --t-start/--t-end, not both")
        try:
            t_start, t_end = parse_window(args.window)
        except ValueError as exc:
            parser.error(str(exc))

    if (t_start is None) ^ (t_end is None):
        parser.error("Both --t-start and --t-end are required when specifying a window")

    if t_start is not None and t_end is not None and t_start >= t_end:
        parser.error(f"Invalid window: start ({t_start}) must be less than end ({t_end})")

    full_time_col = "elapsed_ms" if args.time_unit == "ms" else "elapsed_s"
    csv_start = df[full_time_col].iloc[0]
    csv_end = df[full_time_col].iloc[-1]
    if t_start is not None and t_start > csv_end:
        parser.error(
            f"--t-start ({t_start} {args.time_unit}) is after CSV end "
            f"({csv_end:.3f} {args.time_unit})"
        )

    df = filter_time_window(
        df, t_start=t_start, t_end=t_end, time_unit=args.time_unit
    )
    if df.empty:
        parser.error(
            "No samples in the requested time window. "
            f"CSV spans {csv_start:.3f}–{csv_end:.3f} {args.time_unit}."
        )
    add_relative_time_columns(df)

    if args.relative_time and args.absolute_time:
        parser.error("Use only one of --relative-time or --absolute-time")

    zero_based = args.zero_time and not args.absolute_time
    window_relative = False
    if t_start is not None:
        window_relative = not args.absolute_time
    elif args.relative_time:
        window_relative = True

    if window_relative and t_start is not None:
        df = apply_window_relative_time(df, t_start=t_start, time_unit=args.time_unit)

    window_label = None
    if t_start is not None and t_end is not None:
        window_label = f"window: {t_start:g}–{t_end:g} {args.time_unit}"
        if window_relative:
            window_label += " (relative x-axis)"

    duration_s = time_span_seconds(df)
    print(
        f"CSV time span: {duration_s:.3f} s "
        f"({len(df):,} samples, "
        f"elapsed_us {df['elapsed_us'].iloc[0]:.0f}–{df['elapsed_us'].iloc[-1]:.0f})"
    )

    burst_stats = None
    burst_intervals: list[tuple[float, float]] | None = None
    if not args.no_peak_stats:
        threshold_mibs = gbps_to_mibs(args.peak_threshold_gbps)
        burst_intervals = detect_memory_bursts(
            df,
            threshold_mibs=threshold_mibs,
            min_duration_us=args.peak_min_duration_us,
            min_separation_us=args.peak_min_separation_us,
        )
        burst_stats = compute_burst_stats(
            df,
            burst_intervals,
            threshold_gbps=args.peak_threshold_gbps,
            threshold_mibs=threshold_mibs,
            min_duration_us=args.peak_min_duration_us,
            min_separation_us=args.peak_min_separation_us,
        )
        print(burst_stats.summary())
        print(
            f"  analysis window: {burst_stats.analysis_duration_s:.3f} s  "
            f"({burst_stats.count} bursts / "
            f"{burst_stats.analysis_duration_s:.3f} s)"
        )

    if window_relative and "plot_ms" in df.columns:
        plot_time_col = "plot_ms" if args.time_unit == "ms" else "plot_s"
    else:
        plot_time_col = select_plot_time_column(
            df, args.time_unit, zero_based=zero_based
        )

    title = f"IMC CAS bandwidth — {csv_path.name}"
    plot_csv(
        df,
        title=title,
        output=output_path,
        show=args.show,
        smooth=max(1, args.smooth),
        time_unit=args.time_unit,
        relative_time=window_relative,
        window_label=window_label,
        burst_stats=burst_stats,
        burst_intervals=burst_intervals,
        plot_time_col=plot_time_col,
        duration_s=duration_s,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
