"""Subprocess errors remain visible and cannot block checkpoint export."""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


NOTEBOOK = Path(__file__).resolve().parents[1] / "run_colab.ipynb"


class NotebookWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source = root / "source"
        (source / "followup").mkdir(parents=True)
        (source / "followup/__init__.py").write_text("")
        (source / "followup/runner.py").write_text(
            "import sys\n"
            "print('fixture stdout', flush=True)\n"
            "print('fixture stderr', file=sys.stderr, flush=True)\n"
            "if sys.argv[1] == 'report':\n"
            "    raise ValueError('fixture figure failure')\n")
        run = root / "drive/followup-v1"
        run.mkdir(parents=True)
        self.namespace = {"PYTHON": sys.executable, "SOURCE": source,
                          "RUN": run, "ROOT": root, "RUN_NAME": run.name,
                          "INDEX": root / "index", "SOURCE_REVISION": "f" * 40,
                          "RUN_ENV": {**os.environ, "PYTHONPATH": str(source)},
                          "Path": Path, "subprocess": subprocess}
        cells = {c["id"]: c for c in json.loads(NOTEBOOK.read_text())["cells"]}
        tree = ast.parse(cells["index"]["source"])
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_stage")
        exec(compile(ast.Module(body=[function], type_ignores=[]), "run_stage", "exec"), self.namespace)
        self.download_code = cells["download"]["source"]

    def test_stage_output_is_retained_on_success_and_failure(self):
        with redirect_stdout(io.StringIO()):
            self.namespace["run_stage"]("smoke")
            with self.assertRaisesRegex(RuntimeError, "ValueError: fixture figure failure"):
                self.namespace["run_stage"]("report")
        for stage in ("smoke", "report"):
            log = (self.namespace["RUN"] / f"logs/{stage}.log").read_text()
            self.assertIn("fixture stdout", log)
            self.assertIn("fixture stderr", log)
        self.assertIn("ValueError: fixture figure failure", log)

    def test_export_after_failed_report_does_not_launch_a_stage(self):
        run = self.namespace["RUN"]
        saved = run / "cache/results/saved.json"
        saved.parent.mkdir(parents=True)
        saved.write_text('{"retained": true}')
        (run / "unfinished.tmp").write_text("partial")
        (run / ".run.lock").write_text("")
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                self.namespace["run_stage"]("report")
        downloaded = []
        google, colab = ModuleType("google"), ModuleType("google.colab")
        colab.files = SimpleNamespace(download=downloaded.append)
        google.colab = colab
        with patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
             patch.object(subprocess, "Popen") as process, redirect_stdout(io.StringIO()):
            exec(self.download_code, self.namespace)
            process.assert_not_called()
        self.assertEqual(len(downloaded), 1)
        with zipfile.ZipFile(downloaded[0]) as archive:
            self.assertEqual(archive.read("followup-v1/cache/results/saved.json"), saved.read_bytes())
            self.assertIn("followup-v1/logs/report.log", archive.namelist())
            self.assertNotIn("followup-v1/unfinished.tmp", archive.namelist())
            self.assertNotIn("followup-v1/.run.lock", archive.namelist())
            self.assertIsNone(archive.testzip())


if __name__ == "__main__":
    unittest.main()
