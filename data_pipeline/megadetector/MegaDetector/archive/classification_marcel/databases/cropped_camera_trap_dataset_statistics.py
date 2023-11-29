########
#
# cropped_camera_trap_dataset_statistics.py
#
# Tools for getting dataset statistics. Works for datasets generated with
# make_classification_dataset.py.')
#
########

#%% Imports

import argparse
import json

import numpy as np
import pycocotools.coco


#%% Dataset review

parser = argparse.ArgumentParser(
    'Tools for getting dataset statistics. Works for datasets generated with '
    'the make_classification_dataset.py script.')
parser.add_argument(
    'camera_trap_json', type=str,
    help='Path to json file of the camera trap dataset from LILA.')
parser.add_argument(
    'train_json', type=str,
    help='Path to train.json generated by make_classification_dataset.py')
parser.add_argument(
    'test_json', type=str,
    help='Path to test.json generated by make_classification_dataset.py')
parser.add_argument(
    '--classlist_output', type=str,
    help=('Path to output a list of classes that corresponds to the outputs of '
          'a network trained with the train.json file'))
parser.add_argument(
    '--location_key', type=str, default='location',
    help='Key in camera trap json used to split the dataset into train/test.')

args = parser.parse_args()

CT_JSON = args.camera_trap_json
TRAIN_JSON = args.train_json
TEST_JSON = args.test_json
CLASSLIST_OUTPUT = args.classlist_output
LOCATION_KEY = args.location_key

coco = pycocotools.coco.COCO(CT_JSON)

def print_locations(json_file: str) -> None:
    """
    Args:
        json_file: str, path to COCO-style JSON file
    """
    with open(json_file, 'rt') as fi:
        js = json.load(fi)
    print('Locations used:')
    print(sorted({
        coco.loadImgs([im['original_key']])[0][LOCATION_KEY]
        for im in js['images']
    }))
    #js_keys = ['/'.join(im['file_name'].split('/')[1:])[:-4] for im in js['images']]
    #for tk in js_keys:
    #    assert np.isclose(1, np.sum(detections[tk]['detection_scores'] > 0.5))
    class_to_name = {c['id']: c['name'] for c in js['categories']}
    sorted_class_ids = sorted(class_to_name.keys())
    if CLASSLIST_OUTPUT is not None and json_file == TRAIN_JSON:
        with open(CLASSLIST_OUTPUT, 'wt') as fi:
            fi.write('\n'.join([class_to_name[c] for c in sorted_class_ids]))
    labels = np.array([a['category_id'] for a in js['annotations']])
    print(f'In total {len(class_to_name)} classes and {len(labels)} images.')
    print('Classes with one or more images:', len(set(labels)))
    print('Images per class:')
    print('{:5} {:<15} {:>11}'.format('ID', 'Name', 'Image count'))
    for c in sorted_class_ids:
        name = class_to_name[c]
        count = np.sum(labels == c)
        print('{:5} {:<15} {:>11}'.format(c, name, count))

print('Statistics of the training split: ')
print_locations(TRAIN_JSON)
print('\n\nStatistics of the testing split: ')
print_locations(TEST_JSON)
