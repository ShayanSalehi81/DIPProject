from __future__ import annotations

from pathlib import Path

import pytest

from microq_robustness.config import Settings
from microq_robustness.errors import ConfigurationError


def test_env_file_controls_sample_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("SAMPLE_COUNT=17\nCONCURRENCY=2\n", encoding="utf-8")
    monkeypatch.delenv("SAMPLE_COUNT", raising=False)
    settings = Settings.load(tmp_path)
    assert settings.sample_count == 17
    assert settings.concurrency == 2


def test_os_environment_overrides_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("SAMPLE_COUNT=17\n", encoding="utf-8")
    monkeypatch.setenv("SAMPLE_COUNT", "23")
    assert Settings.load(tmp_path).sample_count == 23


def test_invalid_boolean_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("VERIFY_IMAGES=perhaps\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="VERIFY_IMAGES"):
        Settings.load(tmp_path)

