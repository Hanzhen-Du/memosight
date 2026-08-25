"""密钥/.env 加载测试。用临时 .env，绝不读真实密钥值。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import env as env_mod


def _reset_loaded():
    env_mod._loaded = False


class TestEnvLoader(unittest.TestCase):
    def tearDown(self):
        _reset_loaded()

    def test_loads_from_dotenv_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text("MEMOSIGHT_TEST_VAR=from_dotenv\n")
            with mock.patch.object(env_mod, "DOTENV_PATH", p), \
                 mock.patch.dict(os.environ, {}, clear=True):
                _reset_loaded()
                env_mod.load_env(override=True)
                self.assertEqual(os.environ.get("MEMOSIGHT_TEST_VAR"), "from_dotenv")

    def test_export_takes_precedence_when_not_override(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text("MEMOSIGHT_TEST_VAR=from_dotenv\n")
            with mock.patch.object(env_mod, "DOTENV_PATH", p), \
                 mock.patch.dict(os.environ, {"MEMOSIGHT_TEST_VAR": "from_export"}, clear=True):
                _reset_loaded()
                env_mod.load_env(override=False)
                self.assertEqual(os.environ.get("MEMOSIGHT_TEST_VAR"), "from_export")

    def test_missing_dotenv_is_not_fatal(self):
        with mock.patch.object(env_mod, "DOTENV_PATH", Path("/no/such/.env")), \
             mock.patch.dict(os.environ, {}, clear=True):
            _reset_loaded()
            env_mod.load_env(override=True)  # 不应抛错
            self.assertIsNone(os.environ.get("ANTHROPIC_API_KEY"))

    def test_get_anthropic_key_reads_env(self):
        with mock.patch.object(env_mod, "DOTENV_PATH", Path("/no/such/.env")), \
             mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-xxx"}, clear=True):
            _reset_loaded()
            self.assertEqual(env_mod.get_anthropic_api_key(), "sk-test-xxx")

    def test_require_raises_when_absent(self):
        with mock.patch.object(env_mod, "DOTENV_PATH", Path("/no/such/.env")), \
             mock.patch.dict(os.environ, {}, clear=True):
            _reset_loaded()
            with self.assertRaises(RuntimeError):
                env_mod.require_anthropic_api_key()

    def test_no_hardcoded_key_in_source(self):
        # 防呆：源码里不得出现真实密钥前缀
        src = Path(env_mod.__file__).read_text()
        self.assertNotIn("sk-ant-", src)

    def test_no_real_key_leaked_in_pipeline_source(self):
        # 强守卫：真实 ANTHROPIC/DEEPSEEK 密钥值绝不出现在任何 pipeline 源码里。
        import pipeline
        from pipeline.env import get_anthropic_api_key, get_deepseek_api_key

        pkg_dir = Path(pipeline.__file__).parent
        sources = "\n".join(
            p.read_text(encoding="utf-8") for p in pkg_dir.rglob("*.py")
        )
        for getter in (get_anthropic_api_key, get_deepseek_api_key):
            val = getter()
            if val:  # 有配置密钥时，断言它没被硬编码进源码
                self.assertNotIn(val, sources)
        # 防呆：不得出现明文密钥前缀字面量
        self.assertNotIn("sk-ant-", sources)


if __name__ == "__main__":
    unittest.main()
