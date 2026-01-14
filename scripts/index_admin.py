from __future__ import annotations

import argparse
from pathlib import Path

from legalrag.config import AppConfig
from legalrag.index.registry import IndexRegistry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index registry admin")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List index versions")

    act = sub.add_parser("activate", help="Activate an index version")
    act.add_argument("version", type=str)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig.load()
    registry = IndexRegistry(Path(cfg.paths.index_dir))

    if args.cmd == "list":
        for v in registry.list_versions():
            print(v)
        return

    if args.cmd == "activate":
        registry.activate(args.version)
        print(f"active={args.version}")


if __name__ == "__main__":
    main()
