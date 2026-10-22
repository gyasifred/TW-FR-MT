import pytest

from twfr_mt.cli import main, serve


def test_cli_help_does_not_import_training_dependencies(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "prepare" in capsys.readouterr().out


def test_remote_bind_requires_authentication_or_explicit_consent(monkeypatch):
    monkeypatch.delenv("TWFR_USERNAME", raising=False)
    monkeypatch.delenv("TWFR_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="Remote serving"):
        serve("not-read.json", "models", "cpu", "0.0.0.0", 7860, False, False)


def test_partial_credentials_fail_before_launch(monkeypatch):
    monkeypatch.setenv("TWFR_USERNAME", "operator")
    monkeypatch.delenv("TWFR_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="both"):
        serve("not-read.json", "models", "cpu", "localhost", 7860, False, False)
