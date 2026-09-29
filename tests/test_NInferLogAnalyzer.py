import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import NInferLogAnalyzer as nla  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
TEXT_LOG = os.path.join(FIXTURES, "ninfer-serve-20260101-120000-65536.log")
FAILED_LOG = os.path.join(FIXTURES, "ninfer-serve-20260101-120000-98304.log")
JSONL_LOG = os.path.join(FIXTURES, "session.requests.jsonl")


def read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def analyze_file(path, **kwargs):
    sessions = nla.parse_text(read(path), path)
    started = [s for s in sessions if s.started or s.requests]
    failed = [s for s in sessions if not (s.started or s.requests)]
    return nla.analyze(started or failed, [], dict(nla.DEFAULT_THRESHOLDS), **kwargs)


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = nla.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TextLogParsing(unittest.TestCase):
    def setUp(self):
        self.session = nla.parse_text_log(read(TEXT_LOG), TEXT_LOG)

    def test_startup_fields(self):
        s = self.session
        self.assertTrue(s.started)
        self.assertEqual(s.model_id, "llama")
        self.assertEqual(s.port, 8080)
        self.assertEqual(s.context, 65536)
        self.assertEqual(s.kv_mode, "explicit")
        self.assertEqual(s.kv_pages, 1024)
        self.assertAlmostEqual(s.slack_mib, 100.0)
        self.assertAlmostEqual(s.free_after_startup_mib, 1024.0)
        self.assertAlmostEqual(s.graph_allowance_mib, 82.0)
        self.assertTrue(s.cuda_graph)
        self.assertAlmostEqual(s.load_seconds, 5.0)
        self.assertAlmostEqual(s.weights_gib, 11.86)
        self.assertEqual(s.requested_context, 65536)
        self.assertEqual(s.launch_stamp, "20260101-120000")

    def test_request_fields(self):
        r1 = self.session.requests[1]
        self.assertEqual(r1.protocol, "openai_chat_completions")
        self.assertTrue(r1.stream)
        self.assertEqual(r1.messages, 20)
        self.assertEqual(r1.tools, 11)
        self.assertTrue(r1.tool_history)
        self.assertTrue(r1.thinking)
        self.assertEqual(r1.reasoning_budget, 2048)
        self.assertTrue(r1.preserve_thinking)
        self.assertAlmostEqual(r1.temperature, 1.0)
        self.assertEqual(r1.finish, "tool_calls")
        self.assertEqual(r1.tool_calls, 1)
        self.assertEqual(r1.prompt_tokens, 30000)
        self.assertEqual(r1.cached_tokens, 0)
        self.assertEqual(r1.computed_tokens, 30000)
        self.assertEqual(r1.reuse, "full_reset")
        self.assertAlmostEqual(r1.ttft_s, 18.0)
        self.assertAlmostEqual(r1.decode_tps, 60.0)
        self.assertAlmostEqual(r1.wall_s, 20.0)
        self.assertEqual(r1.spec_backend, "mtp")
        self.assertAlmostEqual(r1.spec_tokens_per_round, 2.8)
        self.assertAlmostEqual(r1.spec_acceptance, 0.6)

    def test_errors_and_throughput(self):
        self.assertEqual(self.session.requests[7].errors, ["client disconnected"])
        self.assertEqual(self.session.requests[9].errors, ["inference request expired while waiting for admission"])
        self.assertFalse(self.session.requests[9].done)
        self.assertEqual(len(self.session.throughput), 2)
        self.assertEqual(self.session.throughput[1].waiting, 1)

    def test_failed_startup(self):
        failed = nla.parse_text_log(read(FAILED_LOG), FAILED_LOG)
        self.assertFalse(failed.started)
        self.assertIn("requires 2190164224 bytes", failed.startup_error)
        self.assertEqual(failed.requested_context, 98304)


class Analysis(unittest.TestCase):
    def setUp(self):
        self.report = analyze_file(TEXT_LOG)
        self.summary = self.report["summary"]

    def test_counts(self):
        s = self.summary
        self.assertEqual(s["requests_done"], 8)
        self.assertEqual(s["requests_with_errors"], 2)
        self.assertEqual(s["requests_in_progress"], 0)
        self.assertEqual(s["finish_counts"], {"tool_calls": 4, "stop_token": 2, "cancelled": 1, "output_limit": 1})
        self.assertEqual(s["client_disconnects"], 1)
        self.assertEqual(s["admission_timeouts"], 1)

    def test_reset_classification(self):
        by_id = {r["request_id"]: r for r in self.report["requests"]}
        self.assertIsNone(by_id[1]["reset_cause"])  # first request of the session
        self.assertEqual(by_id[3]["reset_cause"], "new_conversation")
        self.assertEqual(by_id[4]["reset_cause"], "after_side_request")
        self.assertEqual(by_id[5]["reset_cause"], "interleaved")
        self.assertEqual(self.summary["costly_resets"], 2)
        self.assertAlmostEqual(self.summary["costly_reset_ttft_s"], 46.0)

    def test_decode_outlier(self):
        self.assertEqual(self.summary["decode_outlier_ids"], [6])

    def test_cache_and_thinking(self):
        s = self.summary
        self.assertEqual(s["tokens"]["prompt"], 30000 + 32000 + 20000 + 34000 + 40000 + 41000 + 41500 + 36000)
        self.assertEqual(s["tokens"]["cached"], 30000 + 40000 + 41000 + 34000)
        self.assertEqual(s["budget_hits"], 2)
        self.assertEqual(s["context_capacity"], 65536)
        self.assertEqual(s["max_prompt_tokens"], 41500)

    def test_recommendations(self):
        codes = {r["code"] for r in self.report["recommendations"]}
        for code in ("low_startup_slack", "cache_evicted_by_side_requests", "cache_thrash_interleaved",
                     "decode_outliers", "output_limit_hits", "client_disconnects", "admission_timeouts", "queueing"):
            self.assertIn(code, codes)
        self.assertNotIn("cache_history_rewritten", codes)

    def test_last(self):
        report = analyze_file(TEXT_LOG, last=3)
        self.assertEqual([r["request_id"] for r in report["requests"]], [7, 8, 9])


class JsonlParsing(unittest.TestCase):
    def test_jsonl(self):
        report = analyze_file(JSONL_LOG)
        session = report["sessions"][0]
        self.assertEqual(session["kind"], "jsonl")
        self.assertEqual(session["context"], 131072)
        self.assertEqual(session["speculative"], "mtp-3")
        self.assertAlmostEqual(session["slack_mib"], 52018944 / 1048576)
        by_id = {r["request_id"]: r for r in report["requests"]}
        self.assertEqual(by_id[1]["finish"], "tool_calls")  # stop_token + tool calls, normalized
        self.assertEqual(by_id[2]["finish"], "stop_token")
        self.assertAlmostEqual(by_id[1]["decode_tps"], 80.0)
        self.assertAlmostEqual(by_id[1]["prefill_tps"], 1600.0)
        self.assertAlmostEqual(by_id[1]["spec_acceptance"], 0.5)
        self.assertAlmostEqual(by_id[1]["spec_tokens_per_round"], 2.5)
        self.assertEqual(by_id[1]["messages"], 2)
        self.assertEqual(report["summary"]["context_capacity"], 131072)

    def test_detection(self):
        self.assertTrue(nla.looks_like_jsonl(read(JSONL_LOG)))
        self.assertFalse(nla.looks_like_jsonl(read(TEXT_LOG)))


class Cli(unittest.TestCase):
    def test_default_output(self):
        code, out, _ = run_cli(TEXT_LOG)
        self.assertEqual(code, 0)
        self.assertIn("Server: llama | Context: 65,536 (explicit)", out)
        self.assertIn("Median decode:", out)
        self.assertIn("cache lost on 2 turns", out)

    def test_all_modes_render(self):
        for flags in (["-t"], ["-v"], ["--context-table"], ["--speed-graph"], ["-r"], ["--markdown"]):
            code, out, err = run_cli(TEXT_LOG, *flags)
            self.assertEqual(code, 0, (flags, err))
            self.assertTrue(out.strip(), flags)

    def test_json_output(self):
        code, out, _ = run_cli(TEXT_LOG, "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["schema_version"], nla.SCHEMA_VERSION)
        self.assertEqual(data["summary"]["requests_done"], 8)

    def test_failed_start_exit_code(self):
        code, out, _ = run_cli(FAILED_LOG)
        self.assertEqual(code, 1)
        self.assertIn("Server did not start at 98,304 tokens", out)

    def test_directory_picks_session_and_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(TEXT_LOG, tmp)
            shutil.copy(FAILED_LOG, tmp)
            code, out, _ = run_cli(tmp, "-t")
            self.assertEqual(code, 0)
            self.assertIn("Context: 65,536", out)
            self.assertIn("Attempt at 98,304 tokens", out)

    def test_directory_prefers_jsonl_for_same_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "ninfer-serve-20260101-130000-131072")
            shutil.copy(TEXT_LOG, base + ".log")
            shutil.copy(JSONL_LOG, base + ".requests.jsonl")
            code, out, _ = run_cli(tmp, "--json")
            data = json.loads(out)
            self.assertEqual(len(data["sessions"]), 1)
            self.assertEqual(data["sessions"][0]["kind"], "jsonl")
            self.assertEqual(data["summary"]["requests_done"], 2)
            # Startup fields missing from the JSONL log are filled from the text log.
            self.assertEqual(data["sessions"][0]["endpoint"], "http://127.0.0.1:8080/v1")

    def test_compare(self):
        code, out, _ = run_cli("--compare", TEXT_LOG, JSONL_LOG)
        self.assertEqual(code, 0)
        self.assertIn("Session Comparison", out)
        self.assertIn("65,536", out)
        self.assertIn("131,072", out)

    def test_usage_errors(self):
        code, _, err = run_cli(TEXT_LOG, "--json", "--markdown")
        self.assertEqual(code, 2)
        self.assertIn("either --json or --markdown", err)
        code, _, err = run_cli(os.path.join(FIXTURES, "does-not-exist.log"))
        self.assertEqual(code, 2)

    def test_thresholds_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"slack_warn_mib": 50}, handle)
            code, out, _ = run_cli(TEXT_LOG, "--json", "--thresholds", path)
            codes = {r["code"] for r in json.loads(out)["recommendations"]}
            self.assertNotIn("low_startup_slack", codes)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"bogus": 1}, handle)
            code, _, err = run_cli(TEXT_LOG, "--thresholds", path)
            self.assertEqual(code, 2)
            self.assertIn("Unknown threshold keys", err)


if __name__ == "__main__":
    unittest.main()
