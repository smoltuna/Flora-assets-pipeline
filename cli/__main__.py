"""Flora Asset Pipeline CLI entry point.

Usage:
  python -m cli export --flower-id 1 --output ./exports
  python -m cli export --all --output ./exports
  python -m cli status
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="flora-cli",
        description="Flora Asset Pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # export subcommand
    export_parser = sub.add_parser("export", help="Export flowers to xcassets format")
    export_parser.add_argument("--flower-id", type=int, help="Export a single flower by ID")
    export_parser.add_argument("--all", action="store_true", help="Export all enriched flowers")
    export_parser.add_argument("--output", type=Path, default=Path("./flora_export"), help="Output directory")

    # status subcommand
    sub.add_parser("status", help="Show pipeline status for all flowers")

    args = parser.parse_args()

    if args.command == "export":
        asyncio.run(_export(args))
    elif args.command == "status":
        asyncio.run(_status())


async def _export(args: argparse.Namespace) -> None:
    from cli.export import export_one, export_all
    if args.flower_id:
        await export_one(args.flower_id, args.output)
    elif args.all:
        await export_all(args.output)
    else:
        print("Specify --flower-id or --all", file=sys.stderr)
        sys.exit(1)


async def _status() -> None:
    from database import async_session_factory, create_tables
    from sqlalchemy import select, func
    from models import Flower
    await create_tables()
    async with async_session_factory() as session:
        result = await session.execute(
            select(Flower.status, func.count(Flower.id)).group_by(Flower.status)
        )
        rows = result.all()
    print(f"{'Status':<20} {'Count':>6}")
    print("-" * 28)
    for status, count in sorted(rows):
        print(f"{status:<20} {count:>6}")


if __name__ == "__main__":
    main()
