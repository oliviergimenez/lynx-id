import argparse
import json

from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file


parser = argparse.ArgumentParser()
parser.add_argument('--image_filenames_path', type=str)
parser.add_argument('--output_megadetector', type=str)
args = parser.parse_args()


with open(args.image_filenames_path, 'r') as f:
    image_filenames = json.load(f)

results = load_and_run_detector_batch(
    model_file='/gpfswork/rech/ads/commun/megadetector/md_v5a.0.0.pt',
    image_file_names=image_filenames,
    quiet=True,
    include_image_size=True,
    confidence_threshold=0.5
)  # if there is an error: pip install --upgrade torch torchvision

output = write_results_to_file(results, args.output_megadetector)
