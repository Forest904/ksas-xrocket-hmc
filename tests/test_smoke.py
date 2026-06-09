from __future__ import annotations

import pytest

from ksas_xrocket import __version__
from ksas_xrocket.cli import main


def test_package_exposes_version() -> None:
    assert __version__


def test_cli_version_exits_successfully(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert __version__ in captured.out


def test_placeholder_command_returns_deferred_message(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["prepare"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "reserved for M1" in captured.out
    assert "no data or model work runs in M0" in captured.out
