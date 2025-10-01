import argparse
import asyncio
import logging

from evaluator.config.defaults import VERBOSE
from evaluator.evaluator import Evaluator

if not VERBOSE:
    # in non-verbose mode we want to suppress the excessive output from MCP server and client
    logging.disable(logging.WARNING)


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", help="Path to YAML config file (.yaml/.yml).", required=False)
    p.add_argument("--no-defaults", action="store_true", help="Use only the YAML file (no defaults).")
    args = p.parse_args()

    evaluator = Evaluator(args.config, not args.no_defaults or args.config is None)
    asyncio.run(evaluator.run())
