import torch

from point_e.models.config import MODEL_CONFIGS


def point_e_generate_pcd_from_text(text, num_points=4096):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using Point-E on device:", device)

    print("Creating Base Model...")
    base_name = "base40M-textvec"
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)











