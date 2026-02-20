#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

from cocomo import ComponentType


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write out SASA surface and ENM pairs for a component.",
    )
    p.add_argument(
        "-t",
        "--types",
        default="component_types",
        help="Component types file basename (default: component_types).",
    )
    p.add_argument(
        "-d",
        "--dir",
        default=".",
        help="Directory containing the types file (default: .).",
    )
    p.add_argument(
        "-c",
        "--component",
        required=True,
        help="Component name in the types list (required).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    types = ComponentType.read_list(args.types, dir=args.dir)
    if args.component not in types:
        avail = ", ".join(sorted(types.keys()))
        print(
            f"ERROR: component {args.component} not found in {args.types}. " f"Available: {avail}",
            file=sys.stderr,
        )
        return 2

    ct = types[args.component]
    ct.writeout("sasa", f"{args.component}.surface")
    ct.writeout("enm", f"{args.component}.enmpairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
