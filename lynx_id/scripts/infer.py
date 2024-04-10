# lynx_id/scripts/infer/infer.py
import argparse
import os
import sys
from ..data.dataset import LynxDataset


def create_parser():
    """Create and return the argument parser for the inference script."""
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to image folder (at least one image).')
    parser.add_argument('--input-data',
                        type=str,
                        required=True,
                        help='Path to folder containing images (at least one image) or .csv file (columns: filepath, '
                             'optional columns: date, location).')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output .csv file.')
    parser.add_argument('--embeddings-path',
                        type=str,
                        required=True,
                        help='Path to the file containing the knowledge base of all our known individuals in '
                             '`safetensors` format (embeddings torch.tensor).')
    parser.add_argument('--update-base-knowledge',
                        action=argparse.BooleanOptionalAction,
                        help='Whether or not to update the embeddings file with the new images.')
    return parser


def main(args=None):
    # Example usage of the parsed arguments
    print(f"This is the infer script.")
    print(f"{args.model_path=}")

    print(f"{args.input_data=}")
    # check presence image or csv
    files = os.listdir(args.input_data)
    extensions = ['.jpg', '.jpeg', '.png', '.csv']
    image_csv_found = False
    for file in files:
        if os.path.isfile(os.path.join(args.input_data, file)) and any(file.lower().endswith(ext) for ext in extensions):
            image_csv_found = True
            break
    if not image_csv_found:
        raise RuntimeError(f"No image files found in the directory '{args.input_data}'.")

    print(f"{args.output_path=}")
    print(f"{args.embeddings_path=}")
    print(f"{args.output_path=}")
    print(f"{args.update_base_knowledge=}")

    lynx





if __name__ == '__main__':
    # Direct script execution: Parse arguments from command line
    parser = create_parser()
    args = parser.parse_args()
    main(args)