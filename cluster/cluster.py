import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torchvision import models
from torch.utils.data import DataLoader
from data_pipeline.dataset import collate

from data_pipeline.dataset import LynxDataset
from data_pipeline.transformations_and_augmentations import transforms

# Load our model
model_weights = torch.load("/gpfswork/rech/ads/commun/kg_tests/model_best_0.512.pth")
model = models.resnet50(pretrained=False)
model.fc = nn.Identity()
model.load_state_dict(model_weights)

# Load the dataset
lynxDataset = LynxDataset(
    Path("/gpfsscratch/rech/ads/commun/datasets/extracted/lynx_dataset_france.csv"),
    transform=transforms,
)

dataloader = DataLoader(lynxDataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate)

# Load the csv
data = pd.read_csv("/gpfsscratch/rech/ads/commun/datasets/extracted/lynx_dataset_france.csv")

# Select some individuals (20 here with approximately 10 img/individual)
lynx_id_counts = data['lynx_id'].value_counts()
selected_individuals = lynx_id_counts[90:100].index.tolist()

data_selected_individuals = data[data["lynx_id"].isin(selected_individuals)]
data_selected_individuals = data_selected_individuals.copy()

indices = data_selected_individuals.index.tolist()

# Compute batch
images = []
for individual_index in indices:
    img = lynxDataset[individual_index][0]['image']
    img = img.transpose(2, 0, 1)
    img = torch.FloatTensor(img)
    print("img.shape", img.shape)

    images.append(img)

print("len(images)", len(images))
images = torch.stack(images, dim=0)
print("images.shape", images.shape)

batch_size = 32
images_batch = torch.chunk(images, int(images.shape[0]/32), dim=0)

# Compute embeddings
embeddings_list = []
for batch in images_batch:
    with torch.no_grad():
        tmp = model(images)
        print("tmp.shape", tmp.shape)
        embeddings_list.append(tmp)

embeddings = torch.cat(embeddings_list, dim=0)
print("embeddings.shape", embeddings.shape)

# utils code
lynx_str_to_int = {}
lynx_int_to_str = {}
for index, string in enumerate(selected_individuals):
    lynx_str_to_int[string] = index
    lynx_int_to_str[index] = string

# Dimensionality reduction
tsne = TSNE(n_components=2, perplexity=50)
embeddings_2d = tsne.fit_transform(embeddings)
print(embeddings_2d.shape)

# Append information to the dataframe
data_selected_individuals.reset_index(inplace=True)
data_selected_individuals[['embedding_x', 'embedding_y']] = pd.DataFrame(embeddings_2d)
data_selected_individuals['lynx_id_int'], _ = pd.factorize(data_selected_individuals["lynx_id"])

# Generate the plot
fig, ax = plt.subplots(figsize=(15, 8))

scatter = plt.scatter(data_selected_individuals['embedding_x'], data_selected_individuals['embedding_y'], c=data_selected_individuals['lynx_id_int'])
# ax.scatter(candidates_dots_x, candidates_dots_y, c='red', marker='o', label='New data')

handles, labels = scatter.legend_elements()
handles.append(ax.scatter([], [], c='red', marker='o'))
labels.append('Candidate')

for i in range(data_selected_individuals.shape[0]):
    plt.annotate(data_selected_individuals.iloc[i]['index'], (embeddings_2d[i, 0], embeddings_2d[i, 1]))

labels = [re.search(r'\d+', key).group() + ' - ' + lynx_int_to_str[int(re.search(r'\d+', key).group())] if re.search(r'\d+', key) else key for key in labels]
legend = ax.legend(handles, labels, loc="center right", title="lynx_id",  bbox_to_anchor=(-0.2, 0.5))

plt.colorbar(label="Lynx ID")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter plot of embeddings with colouring by lynx_id")
plt.savefig("embeddings.png", bbox_extra_artists=(legend,), bbox_inches="tight")
