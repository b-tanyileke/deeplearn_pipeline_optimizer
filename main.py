"""
Main file for the Deep Learning Pipeline Optimizer.

Usage:
    python main.py --model cnn --trials 20 (or you choice of trials)
    python main.py --model rnn --trials 15
"""

import argparse
from optimizers.optuna_runner import run_optuna_search

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Pipeline Optimizer with Optuna")
    parser.add_argument(
        "--model", type=str, choices=["cnn", "rnn"], default="cnn",
        help="Model architecture to optimize (cnn or rnn)"
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of Optuna trials to run"
    )

    args = parser.parse_args()

    print(f"Running Optuna optimization for model: {args.model}")
    print(f"Number of trials: {args.trials}")

    run_optuna_search(model_type=args.model, n_trials=args.trials)

if __name__ == "__main__":
    main()
