"""Command-line entrypoint for the KSAS XROCKET HMC project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ksas_xrocket import __version__
from ksas_xrocket.audit import DatasetAuditError, audit_dataset

_PLACEHOLDERS = {
    "prepare": "M1 data placement and provenance",
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

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit raw KSAS CSVs and generate sample manifests.",
    )
    audit_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/KSAS-Dataset"),
        help="Raw KSAS dataset directory containing README.md and movements/.",
    )
    audit_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/samples.csv"),
        help="Output path for the canonical sample manifest.",
    )
    audit_parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("results/audit"),
        help="Output directory for audit summary artifacts.",
    )
    audit_parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/manifests/ksas_provenance.json"),
        help="Output path for dataset provenance metadata.",
    )
    audit_parser.set_defaults(command="audit")

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

    if args.command == "audit":
        try:
            result = audit_dataset(
                raw_dir=args.raw_dir,
                manifest_path=args.manifest,
                audit_dir=args.audit_dir,
                provenance_path=args.provenance,
            )
        except DatasetAuditError as exc:
            print(str(exc))
            return 1

        print("KSAS audit passed.")
        print(f"Manifest: {result.manifest_path}")
        print(f"Summary: {result.summary_path}")
        print(f"Numeric ranges: {result.numeric_ranges_path}")
        print(f"Provenance: {result.provenance_path}")
        return 0

    milestone = _PLACEHOLDERS[args.command]
    print(f"`hmc {args.command}` is reserved for {milestone}; no data or model work runs in M0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
