# lynx_id/scripts/infer/infer.py
import argparse

def main(args=None):
    parser = argparse.ArgumentParser(description="Inference script.")
    
    # Define your arguments here
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model for inference.')
    parser.add_argument('--input-data', type=str, required=True, help='Path to input data for inference.')
    parser.add_argument('--output-path', type=str, required=True, help='Path for saving inference outputs.')

    # Parse the arguments passed to main
    parsed_args = parser.parse_args(args)

    # Example usage of the parsed arguments
    print(f"This is the eval script.")
    print(f"Model path: {parsed_args.model_path}")
    print(f"Input data: {parsed_args.input_data}")
    print(f"Output path: {parsed_args.output_path}")

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

    
