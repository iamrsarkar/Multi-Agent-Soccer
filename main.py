import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent 3v3 Soccer")
    subparsers = parser.add_subparsers(dest="command")

    # Training parser
    train_parser = subparsers.add_parser("train", help="Train the agents")
    train_parser.add_argument("--episodes", type=int, default=5000)
    train_parser.add_argument("--rollout-length", type=int, default=256)
    train_parser.add_argument("--log-dir", type=str, default="results/tensorboard")
    train_parser.add_argument("--checkpoint-dir", type=str, default="models")
    train_parser.add_argument("--save-interval", type=int, default=100)

    # Evaluation parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to the trained model")

    args = parser.parse_args()

    if args.command == "train":
        from training.train_selfplay import train
        train(args)
    elif args.command == "evaluate":
        from evaluation.evaluate_match import evaluate
        evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()