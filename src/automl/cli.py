"""Command-line entry point. Commands are added stage by stage (see docs/PLAN.md)."""

import argparse

from automl import __version__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="automl", description=__doc__)
    parser.add_argument("--version", action="version", version=f"automl {__version__}")
    parser.parse_args(argv)
    parser.print_help()


if __name__ == "__main__":
    main()
