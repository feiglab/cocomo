from __future__ import annotations

import argparse
import json

from .__version__ import __version__
from .cocomo_model import COCOMO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cocomo", description="COCOMO coarse-grained model")
    p.add_argument("--version", action="version", version=__version__)
    sp = p.add_subparsers(dest="cmd")

    sp_sim = sp.add_parser("info", help="Show model info")
    sp_sim.add_argument("--name", default="default", help="Model name/preset")
    sp_sim.set_defaults(func=_cmd_info)
    return p


def _cmd_info(args: argparse.Namespace) -> None:
    c = COCOMO()
    print(json.dumps({"model": c.describe()}, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
