"""`halo knowledge ...` commands: inspect the per-machine knowledge store."""

from __future__ import annotations

import argparse
import json

from rich.console import Console

from halo.sast.knowledge import knowledge_dir, prune_knowledge


def run_knowledge(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="halo knowledge",
        description="Inspect the per-machine SAST knowledge store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prune = sub.add_parser("prune", help="Delete snapshots older than N days.")
    prune.add_argument(
        "--max-age-days",
        type=int,
        default=90,
        help="Delete snapshots older than this many days (default: 90).",
    )
    sub.add_parser("path", help="Print the knowledge store location.")
    args = parser.parse_args(argv)

    console = Console()
    if args.command == "path":
        console.print(str(knowledge_dir()))
        return 0
    removed = prune_knowledge(max_age_days=args.max_age_days)
    console.print(
        json.dumps(
            {"pruned": removed, "count": len(removed), "store": str(knowledge_dir())},
            ensure_ascii=False,
        )
    )
    return 0
