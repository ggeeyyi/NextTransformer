"""`python -m nextt_gym.backends.koala chain RUN_DIR`: the segment driver (chain.py)."""
import argparse

from .chain import chain

parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["chain"]); parser.add_argument("run_dir")
chain(parser.parse_args().run_dir)
