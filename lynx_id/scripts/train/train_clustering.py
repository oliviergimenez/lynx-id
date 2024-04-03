# lynx_id/train.py
import argparse


def main(args):
    parser = argparse.ArgumentParser(description="Run training.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')

    # Parse args passed from __main__.py
    parsed_args = parser.parse_args(args)

    print(f"Training started with epochs={parsed_args.epochs} and batch_size={parsed_args.batch_size}.")


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
