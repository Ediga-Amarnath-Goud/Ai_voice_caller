"""
Unit tests for core.config and main CLI entrypoint.
Validates environment variable loading, Pydantic defaults, driver factory dispatch,
and argument parser configuration.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from core.config import AgentConfig
from main import parse_args, build_driver, build_app


class TestAgentConfig(unittest.TestCase):
    """Test suite for AgentConfig model."""

    def test_default_config_values(self):
        cfg = AgentConfig()
        self.assertEqual(cfg.model_name, "gemini-3.8-live")
        self.assertEqual(cfg.voice_name, "Aoede")
        self.assertEqual(cfg.input_sample_rate, 16000)
        self.assertEqual(cfg.output_sample_rate, 24000)
        self.assertEqual(cfg.telephony_driver, "virtual")
        self.assertEqual(cfg.inbound_policy, "manual_approval")
        self.assertEqual(cfg.port, 8000)
        self.assertIn("Aoede", cfg.system_prompt)

    def test_from_env_with_overrides(self):
        with patch.dict(os.environ, {
            "VOICE_CALLER_DRIVER": "windows",
            "VOICE_CALLER_PORT": "9000",
            "VOICE_CALLER_VOICE": "Puck",
        }):
            cfg = AgentConfig.from_env(port=9500)
            self.assertEqual(cfg.telephony_driver, "windows")
            self.assertEqual(cfg.voice_name, "Puck")
            self.assertEqual(cfg.port, 9500)


class TestMainCLI(unittest.TestCase):
    """Test suite for CLI arguments and component builders."""

    def test_parse_args_defaults(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = parse_args()
            self.assertEqual(args.driver, "virtual")
            self.assertEqual(args.port, 8000)
            self.assertEqual(args.voice, "Aoede")
            self.assertEqual(args.model, "gemini-3.8-live")

    def test_parse_args_custom(self):
        with patch.object(sys, "argv", [
            "main.py", "--driver", "windows", "--port", "8888", "--voice", "Charon", "--log-level", "DEBUG"
        ]):
            args = parse_args()
            self.assertEqual(args.driver, "windows")
            self.assertEqual(args.port, 8888)
            self.assertEqual(args.voice, "Charon")
            self.assertEqual(args.log_level, "DEBUG")

    def test_build_driver_virtual(self):
        with patch.object(sys, "argv", ["main.py", "--driver", "virtual"]):
            args = parse_args()
            driver = build_driver(args)
            self.assertIn("VirtualSimulator", driver.driver_name)

    def test_build_driver_linux(self):
        with patch.object(sys, "argv", ["main.py", "--driver", "linux"]):
            args = parse_args()
            driver = build_driver(args)
            # If LinuxDriver is not available on Windows, it gracefully falls back to virtual
            self.assertTrue("LinuxDriver" in driver.driver_name or "VirtualSimulator" in driver.driver_name)

    def test_build_driver_invalid(self):
        mock_args = MagicMock()
        mock_args.driver = "unsupported_driver"
        with self.assertRaises(ValueError):
            build_driver(mock_args)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-mock"})
    def test_build_app(self):
        with patch.object(sys, "argv", ["main.py", "--driver", "virtual"]):
            args = parse_args()
            app = build_app(args)
            self.assertIsNotNone(app)
            self.assertEqual(app.title, "AI Voice Calling Assistant API")


if __name__ == "__main__":
    unittest.main()
