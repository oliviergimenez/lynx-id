import pandas as pd
import timm
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from lynx_id.utils import dinov2_utils


class EmbeddingModel:
    def __init__(self, device: str, model_path: str = None, base_resnet: bool = False, model_type="resnet", custom_path=None):
        self.device = device
        if model_type == "resnet":
            self.model_path = model_path
            self.base_resnet = base_resnet
    
            self.model = self.load_model()

        elif model_type == "dinov2":
            torch_hub_dir = dinov2_utils.set_torch_hub_dir(custom_path=custom_path)
            model_name = 'dinov2_vitl14_reg'                
            self.model = torch.hub.load('/lustre/fswork/projects/rech/ads/commun/models/facebookresearch_dinov2_main/', model_name, source='local').to(device)

        elif model_type == "megadescriptor":
            self.model = timm.create_model(
                "swin_large_patch4_window12_384",
                pretrained=True,
                # features_only=True,
                num_classes=0,
                pretrained_cfg_overlay={'file': '/lustre/fswork/projects/rech/ads/commun/models/MegaDescriptor-L-384/pytorch_model.bin'}
            ).to("cuda")
    
    
    def load_model(self):
        model_weights = torch.load(self.model_path,  map_location=self.device)
        model = models.resnet50(weights=None)
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
            for i, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc="Compute embeddings")):
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
