import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Assuming $ALL_CCFRWORK is an environment variable that stores the desired path
hub_dir = os.path.join(os.getenv('ALL_CCFRWORK'),'models')

if hub_dir is not None:
    torch.hub.set_dir(hub_dir)
else:
    print("Environment variable 'ALL_CCFRWORK' is not set.")


dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
