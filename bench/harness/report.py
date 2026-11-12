from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

COLUMNS = (
    "backend",
    "device",
    "hardware",
    "parity",
    "max_abs_err",
    "rel_l2_err",
    "ms_median",
    "ms_min",
    "ms_p90",
    "gb_s",
    "gflops",
    "pct_of_peak",
    "status",
)


def latest_result(results_dir: Path) -> Path:
    candidates = sorted(
        [
            path
            for path in results_dir.glob("*.json")
            if path.name != "schema.json" and "mojo_status" not in path.name and not path.name.startswith(("env", "gpu_check"))
        ],
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidates:
        raise SystemExit(f"no result files in {results_dir}; run `just bench-rmsnorm`")
    return candidates[-1]


def load_document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if value == 0.0:
                return "0"
            if abs(value) < 1e-3:
                return f"{value:.2e}"
            return f"{value:.{digits}f}"
        return str(value)
    return str(value)


def row_label(row: dict[str, Any]) -> str:
    backend = row.get("backend", "?")
    status = row.get("status", "?")
    if status != "ok":
        note = row.get("note") or row.get("error") or status
        return f"{backend} ({' '.join(str(note).split())[:56]})"
    return backend


def build_table(rows: Sequence[dict[str, Any]]) -> str:
    header = ["backend", "device", "parity", "max_abs_err", "rel_l2_err", "ms", "GB/s", "% of peak", "status"]
    grid = [header]
    for row in rows:
        parity = "PASS" if row.get("parity_pass") else "FAIL"
        grid.append(
            [
                row_label(row).replace("|", "/"),
                str(row.get("device", "-")),
                parity,
                fmt(row.get("max_abs_err"), 6),
                fmt(row.get("rel_l2_err"), 6),
                fmt(row.get("ms_median")),
                fmt(row.get("gb_s"), 1),
                fmt(row.get("pct_of_peak"), 1),
                str(row.get("status", "?")),
            ]
        )
    widths = [max(len(line[column]) for line in grid) for column in range(len(header))]
    lines = ["| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(grid[0])) + " |"]
    lines.append("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    for entry in grid[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(entry)) + " |")
    return "\n".join(lines)


def group_by_key(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["key"]].append(row)
    for entries in grouped.values():
        entries.sort(key=lambda row: (not row.get("parity_pass"), row.get("ms_median") or float("inf")))
    return dict(grouped)


def render(document: dict[str, Any], only_op: str | None) -> str:
    rows = [row for row in document["rows"] if only_op is None or row.get("op") == only_op]
    peak = document.get("peak", {})
    header = [
        f"# results {document.get('created_utc', '')} tag={document.get('tag', '')}",
        f"device: {peak.get('device')}  peak: {fmt(peak.get('value_gbytes_per_s'), 1)} GB/s ({peak.get('source')})",
        f"torch={document.get('host', {}).get('torch')} triton={document.get('host', {}).get('triton')} python={document.get('host', {}).get('python')}",
        f"threads={document.get('settings', {}).get('threads')} reps={document.get('settings', {}).get('reps')} warmup={document.get('settings', {}).get('warmup')}",
    ]
    body: list[str] = []
    for key, entries in sorted(group_by_key(rows).items()):
        sample = entries[0]
        body.append(f"\n## {key}  dims={sample.get('dims')} dtype={sample.get('dtype')}")
        if sample.get("op") in {"rmsnorm", "softmax"}:
            body.append(f"traffic={sample.get('traffic_bytes', '')} bytes; bandwidth is the fair yardstick for these.")
        body.append(build_table(entries))
        failures = [entry for entry in entries if not entry.get("parity_pass") and entry.get("status") == "ok"]
        for entry in failures:
            body.append(
                f"- FAIL {entry.get('backend')}: max_abs={fmt(entry.get('max_abs_err'), 6)} "
                f"rel_l2={fmt(entry.get('rel_l2_err'), 6)} vs limits atol={entry.get('atol')} "
                f"rtol={entry.get('rtol')} rel_l2={entry.get('rel_l2_limit')}"
            )
    return "\n".join(header + body) + "\n"


def export_csv(document: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS) + ["key", "op", "dims", "note", "error"], extrasaction="ignore")
        writer.writeheader()
        for row in document["rows"]:
            flat = dict(row)
            flat["parity"] = "PASS" if row.get("parity_pass") else "FAIL"
            flat["dims"] = "x".join(str(part) for part in row.get("dims", []))
            writer.writerow(flat)


def native_hardware(document: dict[str, Any]) -> str:
    hardware = document.get("hardware", {})
    if not hardware:
        return "native"
    cores = hardware.get("available_parallelism", "?")
    avx = "avx2" if hardware.get("avx2") else ("avx" if hardware.get("avx") else "scalar")
    return f"cpu x {cores} cores {avx} rust {hardware.get('engine_version', '?')}"


def native_metrics(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", {})
    return {
        "ms_median": result.get("ms_median", summary.get("ms_median")),
        "ms_min": result.get("ms_min", summary.get("ms_min")),
        "ms_p90": result.get("ms_p90", summary.get("ms_p90")),
        "gb_s": result.get("gb_s"),
        "gflops": result.get("gflops"),
    }


def rows_from_native_dir(directory: Path) -> list[dict[str, Any]]:
    from brainlab.spec.config import load_model_spec
    from brainlab.spec.workloads import WORKLOADS

    spec = load_model_spec()
    sized = [workload.with_model(spec.param_count, spec.vocab_size) for workload in WORKLOADS]
    by_key = {workload.key: workload for workload in sized}
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        document = json.loads(text)
        key = document.get("key") or path.stem
        result = document.get("result") or document
        parity = result.get("parity", document.get("parity", {}))
        workload = by_key.get(key)
        if workload is None or workload.op != document.get("op", workload.op):
            continue
        metrics = native_metrics(result)
        ms = metrics["ms_median"]
        traffic = workload.traffic_bytes if workload else 0
        if ms and traffic and metrics["gb_s"] is None:
            metrics["gb_s"] = traffic / (float(ms) * 1e-3) / 1e9
        rows.append(
            {
                "key": key,
                "op": document.get("op", result.get("op") or (workload.op if workload else "?")),
                "dtype": document.get("golden_dtype", document.get("dtype", "?")),
                "dims": list(workload.dims) if workload else [],
                "backend": document.get("backend", "native"),
                "device": "cpu",
                "hardware": native_hardware(document),
                "traffic_bytes": traffic,
                "flops": workload.flops if workload else None,
                "speedup_vs_dot": result.get("speedup_vs_dot"),
                "baseline": result.get("baseline"),
                "status": "ok" if parity else "info",
                "parity_pass": bool(parity.get("parity_pass", False)),
                "max_abs_err": parity.get("max_abs_err"),
                "rel_l2_err": parity.get("rel_l2_err"),
                "atol": parity.get("atol"),
                "rtol": parity.get("rtol"),
                "rel_l2_limit": parity.get("rel_l2_limit"),
                "note": None if parity else result.get("kernel", "no parity block"),
                **metrics,
            }
        )
    return rows


def merge_native(document: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(document)
    existing = {(row.get("key"), row.get("backend"), row.get("device")) for row in document.get("rows", [])}
    added = [row for row in rows if (row["key"], row["backend"], row["device"]) not in existing]
    merged["rows"] = list(document.get("rows", [])) + added
    return merged


def standalone_document(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "created_utc": "",
        "tag": "native-only",
        "peak": {"device": "cpu", "value_gbytes_per_s": None, "source": "cpu rows are not compared against a GPU peak"},
        "host": {},
        "settings": {},
        "rows": list(rows),
    }


def render_profile(document: dict[str, Any]) -> str:
    total = document.get("forward_ms_per_run")
    accounted = document.get("forward_accounted_ms_per_run")
    coverage = (
        ""
        if not total or accounted is None
        else f"  (timers cover {100.0 * accounted / total:.1f}% of the run)"
    )
    lines = [
        f"# profile {document.get('key', '')} batch={document.get('batch')} seq={document.get('sequence')}"
        f" reps={document.get('reps')} threads={document.get('threads')}",
        f"forward_ms_per_run: {fmt(total, 3)}  accounted: {fmt(accounted, 3)}{coverage}",
    ]
    for section, label in (("forward_time_share", "forward"), ("decode", None)):
        if section == "decode":
            block = document.get("decode")
            if not isinstance(block, dict):
                continue
            lines.append(
                f"\ndecode: steps={block.get('steps')} prefix={block.get('prefill_prefix')}"
                f" ms_per_step={fmt(block.get('ms_per_step'), 3)}"
                f" accounted={fmt(block.get('accounted_ms_per_step'), 3)}"
                f" tokens_per_sec={fmt(block.get('tokens_per_sec'), 1)}"
            )
            entries = block.get("time_share", [])
        else:
            lines.append(f"\n## {label} time share")
            entries = document.get(section, [])
        grid = [["op", "ms_per_run", "share_pct", "calls_per_run"]]
        for entry in entries:
            grid.append(
                [
                    str(entry.get("op")),
                    fmt(entry.get("ms_per_run"), 4),
                    fmt(entry.get("share_pct"), 1),
                    fmt(entry.get("calls_per_run"), 2),
                ]
            )
        if len(grid) == 1:
            lines.append("(no entries)")
            continue
        widths = [max(len(row[column]) for row in grid) for column in range(4)]
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(grid[0])) + " |")
        lines.append("|" + "|".join("-" * (width + 2) for width in widths) + "|")
        for row in grid[1:]:
            lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainlab-report")
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--op", default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--merge-dir", type=Path, default=None, help="directory of per-key native JSON to fold into the table")
    parser.add_argument("--profile", type=Path, default=None, help="render a brainlab profile JSON instead of a results table")
    args = parser.parse_args(argv)

    if args.profile:
        print(render_profile(load_document(args.profile)))
        return 0

    native_rows = rows_from_native_dir(args.merge_dir) if args.merge_dir else []
    path = args.file
    if path is None:
        root = Path(__file__).resolve().parents[2]
        if native_rows and not list((root / "bench" / "results").glob("2*.json")):
            document = standalone_document(native_rows)
        else:
            path = latest_result(args.results_dir or (root / "bench" / "results"))
            document = merge_native(load_document(path), native_rows)
    else:
        document = merge_native(load_document(path), native_rows)
    if args.csv:
        export_csv(document, args.csv)
        print(f"wrote {args.csv}")
    if args.json:
        print(json.dumps(document["rows"], indent=2, sort_keys=True))
        return 0
    print(f"source: {path or 'merged native runs'}")
    print(render(document, args.op))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
