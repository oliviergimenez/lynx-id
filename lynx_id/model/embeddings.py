import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

import pandas as pd


class EmbeddingModel:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device

        self.model = self.load_model()

    def load_model(self):
        model_weights = torch.load(self.model_path)
        model = models.resnet50(pretrained=False)
        model.fc = nn.Identity()
        model.load_state_dict(model_weights)
        return model.to(self.device)

    def compute_embeddings(self, dataloader: DataLoader, save_embeddings_path: str = None, save_lynx_id_path: str = None):
        embeddings = None
        lynx_ids = []
        self.model.eval()

        for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            batch_tensor = torch.stack(batch[0]['image']).to(self.device).float()

            if save_lynx_id_path:
                lynx_ids.extend(batch[1]['lynx_id'])

            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)

                if embeddings is None:
                    embeddings = batch_embeddings
                else:
                    embeddings = torch.cat((embeddings, batch_embeddings), dim=0)

        if save_embeddings_path:
            save_file({"embeddings": embeddings}, save_embeddings_path)

        if save_lynx_id_path:
            df_lynx_ids = pd.DataFrame(lynx_ids, columns=['lynx_id'])
            df_lynx_ids.to_csv(save_lynx_id_path)

        return embeddings
