import contextlib
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest

from jetson_edge_vision.cli import main
from jetson_edge_vision.config import AppConfig, load_config
from jetson_edge_vision.logging_utils import JsonFormatter, configure_logging


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(load_config(), AppConfig())

    def test_example(self):
        self.assertEqual(load_config(Path(__file__).resolve().parents[1] / "configs/default.json"), AppConfig())

    def test_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for data in ('[]', '{', '{"extra": 1}', '{"device": "tpu"}',
                         '{"log_level": "verbose"}', '{"device": null}', '{"log_level": []}'):
                with self.subTest(data=data):
                    path.write_text(data)
                    with self.assertRaises(ValueError):
                        load_config(path)
            path.write_text('{"device": "cpu", "log_level": "DEBUG"}')
            self.assertEqual(load_config(path), AppConfig("DEBUG", "cpu"))

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Cannot read configuration"):
                load_config(Path(directory) / "missing.json")


class LoggingTests(unittest.TestCase):
    def test_json_and_exception(self):
        try:
            raise RuntimeError("test failure")
        except RuntimeError:
            import sys
            record = logging.LogRecord("test", logging.ERROR, "", 0, 'hello %s', ('world',), sys.exc_info())
        data = json.loads(JsonFormatter().format(record))
        self.assertEqual(data["message"], "hello world")
        self.assertEqual(data["level"], "ERROR")
        self.assertIn("RuntimeError: test failure", data["exception"])
        self.assertIn("+00:00", data["timestamp"])

    def test_reconfiguration_does_not_duplicate_logs(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            configure_logging()
            configure_logging().info("ready")
        self.assertEqual(len(stream.getvalue().splitlines()), 1)


class CliTests(unittest.TestCase):
    def test_startup(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(main([]), 0)
        self.assertIn("Foundation ready", json.loads(output.getvalue())["message"])

    def test_bad_config_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(main(["--config", str(Path(directory) / "missing")]), 1)
        self.assertEqual(json.loads(output.getvalue())["level"], "ERROR")


if __name__ == "__main__":
    unittest.main()
