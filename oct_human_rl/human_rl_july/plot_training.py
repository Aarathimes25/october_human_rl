"""
plot_training.py — generate every analytics chart.

Usage
    python plot_training.py
    python plot_training.py --log checkpoints/training_log.csv --out results
    python plot_training.py --eval results/evaluation.json

Reads the training log written by train.py and, when present, the evaluation
results written by evaluate.py, and writes PNG charts into the results
directory.  All plotting logic lives in utils/analytics.py — this file is just
the command-line entry point.
"""

import argparse

from utils.analytics import generate_all, DEFAULT_LOG, DEFAULT_EVAL, DEFAULT_OUT


def parse_args():
    p = argparse.ArgumentParser(description="Generate DTS-ZSC analytics charts")
    p.add_argument("--log",  type=str, default=DEFAULT_LOG,
                   help="training log CSV written by train.py")
    p.add_argument("--eval", type=str, default=DEFAULT_EVAL,
                   help="evaluation JSON written by evaluate.py")
    p.add_argument("--out",  type=str, default=DEFAULT_OUT,
                   help="directory to write the charts into")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("DTS-ZSC Performance Analytics")
    print("=" * 70)
    written = generate_all(log_path=args.log, eval_path=args.eval, out_dir=args.out)
    print("=" * 70)
    if written:
        print(f"{len(written)} chart(s) written to {args.out}/")
    else:
        print("Nothing to plot yet. Run train.py, then evaluate.py --baselines.")
    print("=" * 70)


if __name__ == "__main__":
    main()
