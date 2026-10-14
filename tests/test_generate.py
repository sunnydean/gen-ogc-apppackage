"""Offline checks for the public generator and environment configuration."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis
import generate
import semantic


class GenerateTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True))
        self.folder = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.options = {"env_file": self.folder / "absent.env"}
        self.graph = nx.DiGraph()
        self.graph.add_node("main", calls=[])
        self.graph.graph["commands"] = {"tool": "main"}
        self.candidate = {"cwl": {"cwlVersion": "v1.2"}, "job": {"count": 1}}
        self.valid = {"valid": True, "findings": [], "warnings": []}
        self.pipeline = MagicMock()
        self.stages = {}
        for name, result in {
            "analyze_repo": self.graph, "container_metadata": {}, "cluster_graph": [],
            "label_clusters": [], "infer_io_connections": [],
            "generate_cwl": self.candidate, "validate_candidate": self.valid,
        }.items():
            stage = stack.enter_context(patch(f"generate.{name}", return_value=result))
            self.pipeline.attach_mock(stage, name)
            self.stages[name] = stage
        self.gemini = stack.enter_context(patch("generate.gemini", return_value={}))
        for target in ("analysis.urlopen", "semantic.urlopen"):
            stack.enter_context(patch(target, side_effect=AssertionError("No network calls.")))

    def run_generator(self, **options):
        return generate.generate("https://github.com/example/science", "ghcr.io/example/science:1",
                                 **{**self.options, **options})

    def assert_model(self, model):
        ask = self.stages["generate_cwl"].call_args.args[1]
        ask("test prompt")
        self.gemini.assert_called_once_with("test prompt", model=model)

    def test_stage_order_parameters_and_two_yaml_strings_without_writes(self):
        with patch("pathlib.Path.write_text", side_effect=AssertionError("No output files.")):
            cwl, job = self.run_generator(extra_docs="Scientific context.", entrypoints=["module.main"],
                                         distance_threshold=6.0, separate_entrypoints=False,
                                         require_graph=False)
        self.assertEqual([call[0] for call in self.pipeline.mock_calls], list(self.stages))
        self.stages["analyze_repo"].assert_called_once_with(
            "https://github.com/example/science", ["module.main"])
        self.stages["cluster_graph"].assert_called_once_with(
            self.graph, distance_threshold=6.0, separate_entrypoints=False)
        self.stages["validate_candidate"].assert_called_once_with(
            self.candidate, self.graph, require_graph=False)
        self.assertEqual(self.graph.graph["extra_docs"], "Scientific context.")
        self.assertEqual(self.graph.graph["container_image"], "ghcr.io/example/science:1")
        self.assertIsInstance(cwl, str)
        self.assertIsInstance(job, str)
        self.assertEqual(yaml.safe_load(cwl), self.candidate["cwl"])
        self.assertEqual(yaml.safe_load(job), self.candidate["job"])
        self.assert_model("gemini-flash-latest")

    def test_generation_and_validation_errors_feed_the_next_attempt(self):
        fixed = {"cwl": {"cwlVersion": "v1.2", "doc": "Fixed."}, "job": {"count": 2}}
        findings = [{"check": "job", "message": "Correct count."}]
        self.stages["generate_cwl"].side_effect = [ValueError("Malformed JSON."), self.candidate, fixed]
        self.stages["validate_candidate"].side_effect = [
            {"valid": False, "findings": findings}, self.valid]
        cwl, job = self.run_generator(max_attempts=3)
        calls = self.stages["generate_cwl"].call_args_list
        self.assertEqual(len(calls), 3)
        self.assertIsNone(calls[1].args[2])
        self.assertIn("Malformed JSON.", str(calls[1].args[3]))
        self.assertEqual(calls[2].args[2:], (self.candidate, findings))
        self.assertEqual(yaml.safe_load(cwl), fixed["cwl"])
        self.assertEqual(yaml.safe_load(job), fixed["job"])

    def test_exhaustion_and_invalid_attempt_count(self):
        self.stages["validate_candidate"].return_value = {
            "valid": False, "findings": [{"message": "Invalid package."}]}
        with patch("pathlib.Path.write_text", side_effect=AssertionError("No output files.")):
            with self.assertRaisesRegex(RuntimeError, "2 attempts"):
                self.run_generator(max_attempts=2)
        self.assertEqual(self.stages["generate_cwl"].call_count, 2)
        self.stages["analyze_repo"].reset_mock()
        with self.assertRaises(ValueError):
            self.run_generator(max_attempts=0)
        self.stages["analyze_repo"].assert_not_called()

    def test_graph_recommendations_are_logged_and_accepted(self):
        self.stages["validate_candidate"].return_value = {
            **self.valid, "warnings": [{"check": "graph", "message": "Review connection."}]}
        with self.assertLogs("generator", level="WARNING") as logs:
            result = self.run_generator(require_graph=False)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.stages["generate_cwl"].call_count, 1)
        self.assertIn("Review connection.", " ".join(logs.output))

    def test_env_file_preserves_process_values_and_model_override(self):
        env_file = self.folder / ".env"
        env_file.write_text('export GOOGLE_API_KEY="file-key" # comment\nGEMINI_MODEL=file-model\n')
        with patch.dict(os.environ, {"GEMINI_MODEL": "environment-model"}, clear=True):
            self.run_generator(env_file=env_file)
            self.assertEqual(os.environ["GOOGLE_API_KEY"], "file-key")
            self.assertEqual(os.environ["GEMINI_MODEL"], "environment-model")
            self.assert_model("environment-model")
        self.gemini.reset_mock()
        self.run_generator(model="chosen-model")
        self.assert_model("chosen-model")

    def test_missing_key_fails_before_source_fetch(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "API_KEY"):
                self.run_generator()
        self.stages["analyze_repo"].assert_not_called()


class TransportTests(unittest.TestCase):
    def test_gemini_uses_environment_key_with_google_fallback(self):
        response = json.dumps({"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}).encode()
        for env, key in (({"GOOGLE_API_KEY": "google-key"}, "google-key"),
                         ({"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"}, "gemini-key")):
            with self.subTest(env=list(env)), patch.dict(os.environ, env, clear=True):
                with patch("semantic.urlopen", return_value=io.BytesIO(response)) as fetch:
                    self.assertEqual(semantic.gemini("Describe.", "test-model"), {"ok": True})
                request = fetch.call_args.args[0]
                self.assertEqual(request.get_header("X-goog-api-key"), key)
                self.assertNotIn(key, request.full_url)

    def test_github_tokens_are_scoped_to_api_host(self):
        for variable in ("GITHUB_TOKEN", "GH_TOKEN"):
            with self.subTest(variable=variable), patch.dict(os.environ, {variable: "test-token"}, clear=True):
                for host in ("api.github.com", "raw.githubusercontent.com", "api.github.com.example.org"):
                    with patch("analysis.urlopen", return_value=io.BytesIO(b"text")) as fetch:
                        self.assertEqual(analysis.fetch_text(f"https://{host}/resource"), "text")
                    request = fetch.call_args.args[0]
                    self.assertEqual(request.get_header("Authorization"),
                                     "Bearer test-token" if host == "api.github.com" else None)


if __name__ == "__main__":
    unittest.main()
