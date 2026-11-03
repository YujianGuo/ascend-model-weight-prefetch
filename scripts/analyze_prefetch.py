"""Analyze Ascend prefetch evidence; never infer service speedup from trace sums."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from bisect import bisect_right
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


class JsonStream:
    def __init__(self, handle):
        self.handle = handle
        self.buffer = ""
        self.pos = 0
        self.eof = False
        self.decoder = json.JSONDecoder(parse_float=Decimal)

    def more(self):
        chunk = self.handle.read(65536)
        self.buffer = self.buffer[self.pos :] + chunk
        self.pos = 0
        self.eof = not chunk

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ""
            self.more()

    def take(self, token):
        if self.peek() != token:
            raise ValueError(f"Invalid trace JSON: expected {token!r}")
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                # A number may end at a chunk boundary before its remaining digits.
                if end == len(self.buffer) and not self.eof:
                    self.more()
                    continue
                self.pos = end
                return value
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError("Invalid or truncated trace JSON") from exc
                self.more()

    def array(self):
        self.take("[")
        if self.peek() != "]":
            while True:
                yield self.value()
                if self.peek() != ",":
                    break
                self.take(",")
        self.take("]")


def trace_events(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        stream = JsonStream(handle)
        if stream.peek() == "[":
            yield from stream.array()
        else:
            stream.take("{")
            found = False
            if stream.peek() != "}":
                while True:
                    key = stream.value()
                    if not isinstance(key, str):
                        raise ValueError("Trace object keys must be strings")
                    stream.take(":")
                    if key == "traceEvents":
                        if found:
                            raise ValueError("Duplicate traceEvents")
                        found = True
                        yield from stream.array()
                    else:
                        stream.value()
                    if stream.peek() != ",":
                        break
                    stream.take(",")
            stream.take("}")
            if not found:
                raise ValueError("Trace object has no traceEvents array")
        if stream.peek():
            raise ValueError("Unexpected data after trace JSON")


def ns(value):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Non-finite trace timestamp/duration")
    return int(number * 1000)


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative interval")
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(end, result[-1][1])
    return result


def intersections(left, right):
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def length(intervals):
    return sum(end - start for start, end in intervals)


def statistics_of(values):
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    values = sorted(values)
    index = (len(values) - 1) * 0.95
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": values[low] + (values[high] - values[low]) * (index - low),
    }


def numeric(value):
    if value in (None, "", "N/A", "None"):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid nonnegative metric: {value!r}")
    return result


def normalized(value):
    return re.sub(r"\s+", "", value or "").replace('"', "")


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_summary(path, components):
    if not path.exists():
        return {"available": False, "groups": []}
    buckets = defaultdict(list)
    evidence = {}
    matched_components = set()
    unassigned = 0
    rows_count = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Name", "Input Shapes", "Duration(us)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns missing: {required}")
        for row in reader:
            rows_count += 1
            matches = [
                c
                for c in components
                if re.search(c["name_regex"], row["Name"])
                and re.search(
                    c.get("shape_regex", ".*"), normalized(row["Input Shapes"])
                )
            ]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous component mapping for {row['Name']}")
            if not matches:
                unassigned += 1
                continue
            component = matches[0]
            matched_components.add(component["name"])
            signature = tuple(
                normalized(row.get(k))
                for k in (
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Output Formats",
                )
            )
            key = (
                component["name"],
                component.get("stage", "unclassified"),
                row.get("Type") or row["Name"],
                *signature,
            )
            buckets[key].append(row)
            evidence[key] = component.get("mapping_evidence")
    groups = []
    for key, rows in sorted(buckets.items()):
        record = {
            "signature": list(key),
            "rows": len(rows),
            "mapping_evidence": evidence[key],
            "kernel_names": sorted({r["Name"] for r in rows}),
        }
        for label, column in (
            ("duration_us", "Duration(us)"),
            ("wait_us", "Wait Time(us)"),
            ("aic_mte2_us", "aic_mte2_time(us)"),
            ("aiv_mte2_us", "aiv_mte2_time(us)"),
        ):
            values = [numeric(r.get(column)) for r in rows]
            record[label] = statistics_of([v for v in values if v is not None])
        groups.append(record)
    return {
        "available": True,
        "sha256": fingerprint(path),
        "rows": rows_count,
        "unassigned_rows": unassigned,
        "groups": groups,
        "unmatched_components": [
            c["name"] for c in components if c["name"] not in matched_components
        ],
    }


def trace_summary(path, selectors, windows):
    if path is None:
        return {"available": False, "reason": "trace unavailable (not measured zero)"}
    pref = []
    comm = []
    join = []
    dropped = 0
    bounds = union([(ns(a), ns(b)) for a, b in windows]) if windows else None
    for event in trace_events(path):
        if not isinstance(event, dict):
            raise ValueError("Trace array entries must be event objects")
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if re.search(selectors.get("join_regex", r"prefetch_join"), name):
            value = numeric(event.get("dur"))
            in_window = bounds is None or (
                "ts" in event
                and bool(
                    intersections(
                        [(ns(event["ts"]), ns(event["ts"]) + ns(event.get("dur", 0)))],
                        bounds,
                    )
                )
            )
            if value is not None and in_window:
                join.append(value)
        if event.get("cat") in {"cpu_op", "enqueue", "dequeue"}:
            continue
        args = event.get("args", {})
        stream_id = str(
            args.get("Physic Stream Id", args.get("stream id", event.get("tid")))
        )
        selected_kinds = []
        for kind, intervals in (("prefetch", pref), ("communication", comm)):
            pattern = selectors.get(kind + "_regex")
            if not pattern or not re.search(pattern, name):
                continue
            if selectors.get(kind + "_pid") is not None and str(
                event.get("pid")
            ) != str(selectors[kind + "_pid"]):
                continue
            if selectors.get(kind + "_stream") is not None and stream_id != str(
                selectors[kind + "_stream"]
            ):
                continue
            selected_kinds.append(kind)
            if len(selected_kinds) > 1:
                raise ValueError(
                    "One device event matches both prefetch and communication selectors"
                )
            if "ts" not in event or "dur" not in event:
                dropped += 1
                continue
            start, duration = ns(event["ts"]), ns(event["dur"])
            if duration < 0:
                raise ValueError("Negative trace duration")
            intervals.append((start, start + duration))
    if bounds is not None:
        pref = [
            piece for interval in pref for piece in intersections([interval], bounds)
        ]
        comm = [
            piece for interval in comm for piece in intersections([interval], bounds)
        ]
    p_union, c_union = union(pref), union(comm)
    overlap = length(intersections(p_union, c_union))
    # Binary search on merged intervals counts events without counting nested HCCL twice.
    ends = [end for _, end in c_union]
    events_overlapping = 0
    for start, end in pref:
        index = bisect_right(ends, start)
        events_overlapping += index < len(c_union) and c_union[index][0] < end
    return {
        "available": True,
        "sha256": fingerprint(path),
        "scope": "explicit time windows"
        if windows
        else "whole capture; stages not isolated",
        "selectors_configured": bool(
            selectors.get("prefetch_regex") and selectors.get("communication_regex")
        ),
        "mapping_evidence": selectors.get("mapping_evidence"),
        "prefetch_intervals": len(pref),
        "communication_intervals": len(comm),
        "prefetch_event_duration_us": statistics_of([(b - a) / 1000 for a, b in pref]),
        "prefetch_sum_us": length(pref) / 1000,
        "prefetch_union_us": length(p_union) / 1000,
        "communication_union_us": length(c_union) / 1000,
        "overlap_union_us": overlap / 1000,
        "prefetch_overlap_fraction": overlap / length(p_union)
        if length(p_union)
        else None,
        "prefetch_events_with_overlap": events_overlapping,
        "join_cpu_us": statistics_of(join),
        "invalid_events_without_time": dropped,
        "conclusion": "temporal overlap only; target attribution and consumer benefit require source/runtime evidence",
    }


def candidate_check(candidate, require_allreduce):
    rejected = []
    missing = []
    weight = candidate.get("weight", {})
    budget = candidate.get("budget_bytes")
    storage = weight.get("storage_bytes")
    alignment = candidate.get("alignment_bytes", 32)
    if any(type(v) is not int or v <= 0 for v in (budget, storage, alignment)):
        rejected.append(
            "budget_bytes/storage_bytes/alignment_bytes must be positive integers"
        )
    elif budget > storage or budget % alignment:
        rejected.append("budget exceeds bound storage or violates required alignment")
    window = candidate.get("window", {})
    if require_allreduce and window.get("kind") != "allreduce":
        rejected.append("candidate does not satisfy required AllReduce overlap window")
    if window.get("consumer_after_window") is False:
        rejected.append("weight consumer precedes the proposed communication window")
    elif window.get("consumer_after_window") is not True:
        missing.append("consumer order not established")
    routing = candidate.get("routing", "none")
    if routing not in {"none", "exact", "speculative"}:
        rejected.append("routing must be none, exact, or speculative")
    if routing == "exact" and window.get("route_known_before_window") is False:
        rejected.append("exact target route is unavailable before the window")
    elif routing == "exact" and window.get("route_known_before_window") is not True:
        missing.append("route availability not established")
    if routing == "speculative" and not candidate.get("route_hit_rate_evidence"):
        missing.append("speculative prefetch requires measured route hit-rate evidence")
    for label, value in (
        ("post-loader storage binding", weight.get("binding_evidence")),
        ("memory hotspot", candidate.get("memory_evidence")),
        ("communication DAG", window.get("source_evidence")),
        ("launch site", candidate.get("start_site")),
        ("consumer join site", candidate.get("join_site")),
    ):
        if not value:
            missing.append(label + " evidence missing")
    return {
        "name": candidate.get("name", "unnamed"),
        "rejected": rejected,
        "missing": missing,
        "status": "ineligible"
        if rejected
        else "needs_evidence"
        if missing
        else "ready_for_single_point_experiment",
        "evidence_basis": "manifest assertions; not automatically verified against source",
    }


def analyze(baseline, candidate, manifest):
    reports = {}
    for label, root in (("baseline", baseline), ("candidate", candidate)):
        if root is None:
            continue
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"Profile directory does not exist: {root}")
        traces = [root / "trace_view.json", root / "trace_view.json.gz"]
        trace = next((p for p in traces if p.exists()), None)
        if trace is None and not (root / "kernel_details.csv").exists():
            raise ValueError(f"No supported trace or kernel_details.csv in {root}")
        reports[label] = {
            "root": str(root.resolve()),
            "kernels": kernel_summary(
                root / "kernel_details.csv", manifest.get("components", [])
            ),
            "trace": trace_summary(
                trace,
                manifest.get("trace", {}),
                manifest.get("runs", {}).get(label, {}).get("trace_windows_us", []),
            ),
        }
    comparisons = []
    if candidate is not None:
        left = {
            tuple(g["signature"]): g for g in reports["baseline"]["kernels"]["groups"]
        }
        right = {
            tuple(g["signature"]): g for g in reports["candidate"]["kernels"]["groups"]
        }
        for signature in sorted(left.keys() | right.keys()):
            a, b = left.get(signature), right.get(signature)
            signature_complete = all(
                value and value.upper() not in {"N/A", "NONE"}
                for value in signature[3:]
            )
            matched = bool(a and b and signature_complete)
            av = a["duration_us"]["p50"] if a else None
            bv = b["duration_us"]["p50"] if b else None
            comparisons.append(
                {
                    "signature": list(signature),
                    "baseline": a,
                    "candidate": b,
                    "matched_shape_dtype_format": matched,
                    "signature_metadata_complete": signature_complete,
                    "duration_p50_change_pct": (bv / av - 1) * 100
                    if matched and av and bv is not None
                    else None,
                }
            )
    invariants = (
        "model_revision",
        "weight_revision",
        "tp",
        "ep",
        "pp",
        "graph_mode",
        "quantization",
        "workload_hash",
        "source_diff_evidence",
    )
    metadata_issues = []
    if candidate is not None:
        a, b = (manifest.get("runs", {}).get(k, {}) for k in ("baseline", "candidate"))
        for key in invariants:
            if a.get(key) is None or b.get(key) is None:
                metadata_issues.append(key + ": missing")
            elif a[key] != b[key]:
                metadata_issues.append(key + ": differs")
    return {
        "schema_version": 1,
        "model": manifest.get("model", {}),
        "runs": reports,
        "comparison_metadata_issues": metadata_issues,
        "kernel_comparisons": comparisons,
        "candidates": [
            candidate_check(c, manifest.get("require_allreduce", True))
            for c in manifest.get("candidates", [])
        ],
        "service_acceptance": "not evaluated by this profiler analyzer",
        "limitations": [
            "A matched signature is descriptive, not causal proof; expert IDs/counts and cache state can still differ.",
            "Unmatched shapes/layouts/dtypes remain separate; missing trace/metrics are unknown.",
            "MTE2 duration is an overlap indicator, not exposed load latency or service TPOT.",
            "Trace event overlap does not prove legal ordering, cache hits, or consumer improvement.",
        ],
    }


def markdown(result):
    lines = [
        "# Weight prefetch evidence",
        "",
        "Service acceptance: not evaluated.",
        "",
        "## Trace intervals",
        "",
        "| Run | Prefetch intervals | Prefetch union us | Overlap us | Fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    scopes = []
    for label, run in result["runs"].items():
        t = run["trace"]
        if t["available"]:
            ratio = t["prefetch_overlap_fraction"]
            ratio_text = f"{ratio:.2%}" if ratio is not None else "unknown"
            lines.append(
                f"| {label} | {t['prefetch_intervals']} | {t['prefetch_union_us']:.3f} | {t['overlap_union_us']:.3f} | {ratio_text} |"
            )
            scopes.append(
                f"{label} scope: {t['scope']}; event mapping: {t['mapping_evidence'] or 'unverified'}."
            )
        else:
            scopes.append(f"{label}: trace unavailable; no measured zero inferred.")
    lines.extend(["", *scopes])
    for label, run in result["runs"].items():
        k = run["kernels"]
        if k["available"]:
            lines.extend(
                [
                    "",
                    f"{label}: {k['rows']} CSV rows; {k['unassigned_rows']} unassigned; "
                    f"unmatched components: {', '.join(k['unmatched_components']) or 'none'}.",
                ]
            )
            if not k["groups"]:
                lines.append(
                    "No kernel component matched; review source/name/shape mapping before drawing conclusions."
                )
    lines.extend(["", "## Comparison constraints", ""])
    lines.extend(
        "- " + s
        for s in result["comparison_metadata_issues"]
        or ["No metadata mismatch declared; verify assertions against runtime."]
    )
    lines.extend(
        [
            "",
            "## Kernel signatures",
            "",
            "| Component / stage / input shape | Matched | Baseline n | Candidate n | P50 change % |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for c in result["kernel_comparisons"]:
        sig = c["signature"]
        title = " / ".join((sig[0], sig[1], sig[3])).replace("|", "\\|")
        change = c["duration_p50_change_pct"]
        value = f"{change:+.3f}" if change is not None else "unknown"
        lines.append(
            f"| {title} | {c['matched_shape_dtype_format']} | {(c['baseline'] or {}).get('rows', 0)} | {(c['candidate'] or {}).get('rows', 0)} | {value} |"
        )
    lines.extend(["", "## Candidate readiness", ""])
    for c in result["candidates"]:
        lines.append(
            f"- {c['name']}: {c['status']}; " + "; ".join(c["rejected"] + c["missing"])
        )
    lines.extend(["", "## Evidence limits", ""])
    lines.extend("- " + item for item in result["limitations"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    destinations = [args.output] + ([args.markdown] if args.markdown else [])
    if len({p.resolve() for p in destinations}) != len(destinations):
        parser.error("output paths must differ")
    inputs = [args.manifest]
    for root in (args.baseline, args.candidate):
        if root:
            inputs.extend(
                root / n
                for n in ("trace_view.json", "trace_view.json.gz", "kernel_details.csv")
            )
    if any(p.resolve() == d.resolve() for p in inputs for d in destinations):
        parser.error("refusing to overwrite an input file")
    for destination in destinations:
        if destination.exists():
            parser.error(f"output already exists: {destination}; choose a new path")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        result = analyze(args.baseline, args.candidate, manifest)
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if args.markdown:
            args.markdown.write_text(markdown(result), encoding="utf-8")
    except (ValueError, OSError, InvalidOperation, KeyError, TypeError) as exc:
        parser.exit(2, f"analysis failed: {exc}\n")


if __name__ == "__main__":
    main()

