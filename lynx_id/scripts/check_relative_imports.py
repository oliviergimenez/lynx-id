# lynx_id/scripts/infer/infer.py
import argparse
import sys

from ..data import transforms as transforms_from_init
from ..data.transformations_and_augmentations import transforms as transforms_from_file


def create_parser():
    """Create and return the argument parser for the inference script."""
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model for inference.')
    parser.add_argument('--input-data', type=str, required=True, help='Path to input data for inference.')
    parser.add_argument('--output-path', type=str, required=True, help='Path for saving inference outputs.')
    return parser


def main(args=None):
    print("Running train_triplets with arguments:", args)

    # Example usage of the parsed arguments
    print(f"This is the infer script.")
    print(f"Model path: {args.model_path}")
    print(f"Input data: {args.input_data}")
    print(f"Output path: {args.output_path}")

    print(transforms_from_file)


if __name__ == '__main__':
    # Direct script execution: Parse arguments from command line
    parser = create_parser()
    args = parser.parse_args()
    main(args)
