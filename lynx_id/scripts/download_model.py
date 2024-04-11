import argparse

from huggingface_hub import hf_hub_download
from huggingface_hub.utils._errors import LocalEntryNotFoundError
from requests.exceptions import HTTPError, ConnectionError
from urllib3.exceptions import MaxRetryError, NewConnectionError


def create_parser():
    """Create and return the argument parser for the embedding model download script."""
    parser = argparse.ArgumentParser(description="Embedding model download script")
    parser.add_argument('--download-path', type=str, required=True, help='Path where the model will be downloaded.')
    parser.add_argument('--hf-repo-id', type=str, default="tbetton/test_lynx_id", help='Path to HuggingFace repo-id.')
    parser.add_argument('--hf-filename', type=str, default="model_best_0.512.pth", help='Name of the file to be '
                                                                                        'downloaded from the repo-id.')
    return parser


def main(args):
    # Example usage of the parsed arguments
    print(f"This is the download script.")
    print(f"{args.download_path=}")
    print(f"{args.hf_repo_id=}")
    print(f"{args.hf_filename=}")

    return download_model(args)


def download_model(args):
    """Download model weights from HuggingFace"""
    try:
        path = hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=args.hf_filename,
            local_dir=args.download_path,
            local_dir_use_symlinks=False
        )
        print(f"The model has been downloaded to the following path: {path}")
    except (ConnectionError, NewConnectionError, MaxRetryError, LocalEntryNotFoundError) as e:
        error_message = "You don't have Internet access. If you are using Jean-Zay, use a frontal to download the " \
                        "model. You are probably currently using a computing node without Internet access."
        raise ConnectionError(error_message)
    except HTTPError as e:
        if e.response.status_code in {404, 401}:
            raise HTTPError("The URL does not seem to exist. Check HuggingFace to see if the `repo_id` and `filename` "
                            "exist.")
        else:
            raise HTTPError("An HTTP error has occurred.")


if __name__ == '__main__':
    # Direct script execution: Parse arguments from command line
    parser = create_parser()
    args = parser.parse_args()
    main(args)
