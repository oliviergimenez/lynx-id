import torch


def collate_single(batch):
    if not batch:
        return {}, {}

    # Sample the first item to get the keys
    first_input, first_output = batch[0]
    batched_input_dict = {key: [] for key in first_input.keys()}
    batched_output_dict = {key: [] for key in first_output.keys()}

    # Iterate over each item in the batch
    for input_dict, output_dict in batch:
        # Append data from input and output dictionaries
        for key in input_dict:
            batched_input_dict[key].append(input_dict[key])
        for key in output_dict:
            batched_output_dict[key].append(output_dict[key])

    return batched_input_dict, batched_output_dict


def collate(batch):
    # Old style, to be removed
    # Initialize lists to gather all elements for each key
    images = []
    sources = []
    patterns = []
    dates = []
    locations = []
    image_numbers = []
    lynx_ids = []

    # Iterate over each item in the batch
    for input_dict, output_dict in batch:
        # Append data from input dictionary
        images.append(input_dict['image'])  # List of images
        sources.append(input_dict['source'])
        patterns.append(input_dict['pattern'])
        dates.append(input_dict['date'])
        locations.append(input_dict['location'])
        image_numbers.append(input_dict['image_number'])

        # Append data from output dictionary
        lynx_ids.append(output_dict['lynx_id'])

    # Construct the batched input and output dictionaries
    batched_input_dict = {
        'images': images,
        # conversion to array not possible because as image size varies
        'sources': sources,
        'patterns': patterns,
        'dates': dates,
        'locations': locations,
        'image_numbers': image_numbers
    }

    batched_output_dict = {
        'lynx_ids': lynx_ids
    }

    return batched_input_dict, batched_output_dict


def collate_triplet_old(batch):
    if not batch:
        return {}

    # Initialize nested dictionaries for the batch
    batched_data = {
        'anchor': {'input': [], 'output': []},
        'positive': {'input': [], 'output': []},
        'negative': {'input': [], 'output': []}
    }

    # Iterate over each triplet in the batch
    for triplet in batch:
        for key in ['anchor', 'positive', 'negative']:
            for subkey in ['input', 'output']:
                batched_data[key][subkey].append(triplet[key][subkey])

    return batched_data


def collate_triplet(batch):
    if not batch:
        return {}

    # Initialize nested dictionaries for the batch
    batched_data = {
        'anchor': {'input': {}, 'output': {}},
        'positive': {'input': {}, 'output': {}},
        'negative': {'input': {}, 'output': {}}
    }

    # Iterate over each triplet in the batch
    for triplet in batch:
        for key in ['anchor', 'positive', 'negative']:
            for subkey in ['input', 'output']:
                for feature_key, feature_value in triplet[key][subkey].items():
                    if feature_key not in batched_data[key][subkey]:
                        batched_data[key][subkey][feature_key] = []
                    batched_data[key][subkey][feature_key].append(feature_value)

    # Post-process features if necessary (e.g., stacking 'image' tensors)
    for key in ['anchor', 'positive', 'negative']:
        if 'image' in batched_data[key]['input']:
            images = batched_data[key]['input']['image']
            batched_data[key]['input']['image'] = torch.stack(images)
        # Additional post-processing for other features can be added here
    return batched_data
