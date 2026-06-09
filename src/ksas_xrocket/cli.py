"""Command-line entrypoint for the KSAS XROCKET HMC project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ksas_xrocket import __version__

_PLACEHOLDERS = {
    "prepare": "M1 data placement and provenance",
    "audit": "M1 data audit and data dictionary",
    "baseline": "M2 preprocessing and baseline models",
    "train": "M3 XROCKET model training",
    "explain": "M4-M6 explanation analyses",
    "figures": "M4-M8 report figure generation",
    "report": "M8 technical report build",
    "reproduce": "M8 full reproduction workflow",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI parser."""
    parser = argparse.ArgumentParser(
        prog="hmc",
        description="KSAS XROCKET human motion computing workflow commands.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")
    for command, milestone in _PLACEHOLDERS.items():
        subparser = subparsers.add_parser(command, help=f"Placeholder for {milestone}.")
        subparser.set_defaults(command=command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    milestone = _PLACEHOLDERS[args.command]
    print(f"`hmc {args.command}` is reserved for {milestone}; no data or model work runs in M0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
