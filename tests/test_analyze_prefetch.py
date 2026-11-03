import csv
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_prefetch.py"
SPEC = importlib.util.spec_from_file_location("prefetch_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class PrefetchAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def trace(self, events, name="trace_view.json", wrapped=False):
        path = self.root / name
        text = json.dumps(
            {"metadata": "sample", "traceEvents": events} if wrapped else events
        )
        if name.endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(text)
        else:
            path.write_text(text, encoding="utf-8")
        return path

    def event(self, name, start, duration, **kwargs):
        return {
            "ph": "X",
            "name": name,
            "ts": str(start),
            "dur": str(duration),
            **kwargs,
        }

    def csv(self, path, shape="32,256;256,256,2048", dtype="BFLOAT16", values=(10, 20)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "Name",
                    "Type",
                    "Input Shapes",
                    "Input Data Types",
                    "Input Formats",
                    "Output Shapes",
                    "Output Data Types",
                    "Output Formats",
                    "Duration(us)",
                    "aic_mte2_time(us)",
                ],
            )
            writer.writeheader()
            for duration in values:
                writer.writerow(
                    {
                        "Name": "GroupedMatmul_hash",
                        "Type": "GroupedMatmul",
                        "Input Shapes": shape,
                        "Input Data Types": dtype,
                        "Input Formats": "ND",
                        "Output Shapes": "32,2048",
                        "Output Data Types": dtype,
                        "Output Formats": "ND",
                        "Duration(us)": duration,
                        "aic_mte2_time(us)": "N/A",
                    }
                )

    def test_stream_list_wrapped_and_gzip(self):
        events = [self.event("CMO", 10, 2)]
        self.assertEqual(list(analysis.trace_events(self.trace(events))), events)
        self.assertEqual(
            list(analysis.trace_events(self.trace(events, "wrapped.json", True))),
            events,
        )
        self.assertEqual(
            list(analysis.trace_events(self.trace(events, "wrapped.json.gz", True))),
            events,
        )

    def test_large_event_and_chunk_boundaries(self):
        events = [
            self.event("CMO", 10, 2, args={"text": "测" * 70000}),
            self.event("x", 12, 3),
        ]
        self.assertEqual(
            list(analysis.trace_events(self.trace(events, wrapped=True))), events
        )

    def test_invalid_and_truncated_json(self):
        for text in ("[", '[{"ph":"X"}', "[]trailing", "[{},]", '{"metadata":1}'):
            path = self.root / "bad.json"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                list(analysis.trace_events(path))

    def test_absolute_timestamp_precision(self):
        self.assertEqual(
            analysis.ns("1791775424750280.841") - analysis.ns("1791775424750280.741"),
            100,
        )

    def test_overlap_uses_union_and_excludes_cpu_duplicates(self):
        events = [
            self.event("CMO", 10, 20),
            self.event("CMO", 10, 20),
            self.event("hcom_allReduce_1", 15, 10),
            self.event("hcom_allReduce_2", 16, 8),
            self.event("hcom_allReduce_cpu", 10, 20, cat="cpu_op"),
        ]
        result = analysis.trace_summary(
            self.trace(events),
            {"prefetch_regex": "^CMO$", "communication_regex": "^hcom_allReduce_"},
            [],
        )
        self.assertEqual(result["overlap_union_us"], 10)
        self.assertEqual(result["prefetch_union_us"], 20)
        self.assertEqual(result["prefetch_sum_us"], 40)
        self.assertEqual(result["prefetch_events_with_overlap"], 2)
        self.assertEqual(result["prefetch_overlap_fraction"], 0.5)

    def test_stream_filter_and_explicit_windows(self):
        events = [
            self.event("CMO", 10, 20, tid=11),
            self.event("AllReduce", 10, 20, tid=12),
            self.event("AllReduce", 100, 20, tid=42),
        ]
        selectors = {
            "prefetch_regex": "^CMO$",
            "communication_regex": "^AllReduce$",
            "communication_stream": 12,
        }
        result = analysis.trace_summary(self.trace(events), selectors, [["15", "25"]])
        self.assertEqual(result["overlap_union_us"], 10)
        self.assertEqual(result["communication_intervals"], 1)

    def test_join_stats_respect_selected_windows(self):
        events = [
            self.event("prefetch_join", 10, 2, cat="cpu_op"),
            self.event("prefetch_join", 100, 50, cat="cpu_op"),
        ]
        result = analysis.trace_summary(self.trace(events), {}, [["0", "20"]])
        self.assertEqual(result["join_cpu_us"]["n"], 1)
        self.assertEqual(result["join_cpu_us"]["p50"], 2)

    def test_same_event_cannot_prove_self_overlap(self):
        selectors = {"prefetch_regex": "CMO", "communication_regex": "CMO"}
        with self.assertRaises(ValueError):
            analysis.trace_summary(
                self.trace([self.event("CMO", 10, 2)]), selectors, []
            )

    def test_missing_trace_is_unknown_and_empty_trace_is_zero(self):
        self.assertFalse(analysis.trace_summary(None, {}, [])["available"])
        result = analysis.trace_summary(self.trace([]), {}, [])
        self.assertTrue(result["available"])
        self.assertEqual(result["prefetch_intervals"], 0)
        self.assertIsNone(result["prefetch_overlap_fraction"])

    def test_negative_and_nan_metrics_are_rejected(self):
        for value in ("-1", "nan", "inf"):
            with self.assertRaises(ValueError):
                analysis.numeric(value)

    def test_shape_dtype_mismatch_stays_unmatched(self):
        base, cand = self.root / "base", self.root / "candidate"
        self.csv(base / "kernel_details.csv", "256,512;128,512,2048")
        self.csv(cand / "kernel_details.csv", "256,256;256,256,2048")
        manifest = {
            "components": [
                {"name": "gmm2", "name_regex": "GroupedMatmul", "stage": "decode"}
            ]
        }
        result = analysis.analyze(base, cand, manifest)
        self.assertEqual(len(result["kernel_comparisons"]), 2)
        self.assertTrue(
            all(
                not c["matched_shape_dtype_format"]
                for c in result["kernel_comparisons"]
            )
        )
        self.csv(cand / "kernel_details.csv", "256,512;128,512,2048", "INT8")
        result = analysis.analyze(base, cand, manifest)
        self.assertTrue(
            all(
                not c["matched_shape_dtype_format"]
                for c in result["kernel_comparisons"]
            )
        )

    def test_matched_comparison_and_missing_metric(self):
        base, cand = self.root / "base", self.root / "candidate"
        self.csv(base / "kernel_details.csv", values=(10, 20))
        self.csv(cand / "kernel_details.csv", values=(5, 10))
        manifest = {"components": [{"name": "gmm2", "name_regex": "GroupedMatmul"}]}
        result = analysis.analyze(base, cand, manifest)
        c = result["kernel_comparisons"][0]
        self.assertEqual(c["duration_p50_change_pct"], -50)
        self.assertIsNone(c["candidate"]["aic_mte2_us"]["mean"])
        self.assertTrue(result["comparison_metadata_issues"])
        self.assertIn("unclassified", c["signature"])

    def test_missing_dtype_cannot_generate_speedup(self):
        base, cand = self.root / "base", self.root / "candidate"
        self.csv(base / "kernel_details.csv", dtype="", values=(10, 20))
        self.csv(cand / "kernel_details.csv", dtype="", values=(5, 10))
        result = analysis.analyze(
            base, cand, {"components": [{"name": "gmm2", "name_regex": "Group"}]}
        )
        comparison = result["kernel_comparisons"][0]
        self.assertFalse(comparison["matched_shape_dtype_format"])
        self.assertIsNone(comparison["duration_p50_change_pct"])

    def test_ambiguous_component_mapping_rejected(self):
        path = self.root / "kernel_details.csv"
        self.csv(path)
        with self.assertRaises(ValueError):
            analysis.kernel_summary(
                path,
                [
                    {"name": "a", "name_regex": "Group"},
                    {"name": "b", "name_regex": "Group"},
                ],
            )

    def test_unmatched_component_is_explicit(self):
        path = self.root / "kernel_details.csv"
        self.csv(path)
        result = analysis.kernel_summary(
            path, [{"name": "gmm", "name_regex": "^different_export_name$"}]
        )
        self.assertEqual(result["unmatched_components"], ["gmm"])
        self.assertEqual(result["unassigned_rows"], 2)
        self.assertEqual(result["groups"], [])

    def test_dense_model_mapping_has_no_qwen_shape_dependency(self):
        path = self.root / "kernel_details.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["Name", "Type", "Input Shapes", "Duration(us)"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "Name": "MatMulV2_dense",
                    "Type": "MatMulV2",
                    "Input Shapes": "8,4096;6144,4096",
                    "Duration(us)": 42,
                }
            )
        result = analysis.kernel_summary(
            path,
            [
                {
                    "name": "dense_qkv",
                    "stage": "decode",
                    "name_regex": "MatMulV2",
                    "shape_regex": "^8,4096;",
                }
            ],
        )
        self.assertEqual(result["groups"][0]["signature"][0], "dense_qkv")
        self.assertEqual(result["groups"][0]["duration_us"]["p50"], 42)

    def test_exact_future_routes_rejected(self):
        candidate = {
            "name": "future_w2",
            "weight": {"storage_bytes": 1024},
            "budget_bytes": 512,
            "routing": "exact",
            "window": {
                "kind": "allreduce",
                "consumer_after_window": True,
                "route_known_before_window": False,
            },
        }
        self.assertEqual(
            analysis.candidate_check(candidate, True)["status"], "ineligible"
        )

    def test_consumer_before_allreduce_and_invalid_budget_rejected(self):
        for budget in (2048, 17, -1, True):
            candidate = {
                "weight": {"storage_bytes": 1024},
                "budget_bytes": budget,
                "window": {"kind": "allreduce", "consumer_after_window": False},
            }
            result = analysis.candidate_check(candidate, True)
            self.assertEqual(result["status"], "ineligible")

    def test_valid_candidate_is_only_ready_for_experiment(self):
        candidate = {
            "name": "qkv",
            "budget_bytes": 1024,
            "weight": {"storage_bytes": 2048, "binding_evidence": "post-loader audit"},
            "window": {
                "kind": "allreduce",
                "consumer_after_window": True,
                "source_evidence": "DAG",
            },
            "memory_evidence": "matched profile",
            "start_site": "before allreduce",
            "join_site": "before qkv",
        }
        self.assertEqual(
            analysis.candidate_check(candidate, True)["status"],
            "ready_for_single_point_experiment",
        )

    def test_cli_report_and_no_overwrite(self):
        self.csv(self.root / "base" / "kernel_details.csv")
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps({"components": [{"name": "gmm2", "name_regex": "Group"}]}),
            encoding="utf-8",
        )
        output = self.root / "output.json"
        report = self.root / "report.md"
        command = [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(self.root / "base"),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--markdown",
            str(report),
        ]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn(
            "not evaluated", json.loads(output.read_text())["service_acceptance"]
        )
        original = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_absent_profile_rejected(self):
        with self.assertRaises(ValueError):
            analysis.analyze(self.root / "missing", None, {})

    def test_cli_cannot_overwrite_manifest(self):
        self.csv(self.root / "base" / "kernel_details.csv")
        manifest = self.root / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--baseline",
                str(self.root / "base"),
                "--manifest",
                str(manifest),
                "--output",
                str(manifest),
            ],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(manifest.read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
