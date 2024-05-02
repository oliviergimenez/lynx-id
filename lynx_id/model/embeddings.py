import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

import pandas as pd


class EmbeddingModel:
    def __init__(self, model_path: str, device: str, base_resnet: bool = False):
        self.model_path = model_path
        self.device = device
        self.base_resnet = base_resnet

        self.model = self.load_model()

    def load_model(self):
        model_weights = torch.load(self.model_path)
        model = models.resnet50(pretrained=False)
        if not self.base_resnet:
            model.fc = nn.Identity()
        model.load_state_dict(model_weights)
        if self.base_resnet:
            model.fc = nn.Identity()
        return model.to(self.device)

    def compute_embeddings(self, dataloader: DataLoader, save_embeddings_path: str = None, save_lynx_infos_path: str = None):
        embeddings = None
        lynx_ids = []
        dates = []
        filepaths = []
        locations = []
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
                batch_tensor = torch.stack(batch[0]['image']).to(self.device).float()

                if save_lynx_infos_path:
                    lynx_ids.extend(batch[1]['lynx_id'])
                    dates.extend(batch[0]['date'])
                    filepaths.extend(batch[0]['filepath'])
                    locations.extend(batch[0]['location'])

                batch_embeddings = self.model(batch_tensor)

                if embeddings is None:
                    embeddings = batch_embeddings
                else:
                    embeddings = torch.cat((embeddings, batch_embeddings), dim=0)

        if save_embeddings_path:  # safetensors format
            save_file({"embeddings": embeddings}, save_embeddings_path)

        if save_lynx_infos_path:  # csv format
            df_lynx_infos = pd.DataFrame({
                "filepath": filepaths,
                "lynx_id": lynx_ids,
                "date": dates,
                "location": locations
            })
            df_lynx_infos.to_csv(save_lynx_infos_path, index=False)

        return embeddings
