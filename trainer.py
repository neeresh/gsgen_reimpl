import torch.nn as nn
from tqdm import tqdm

from data import CameraPoseProvider
from data.sit3d import SingleViewCameraPoseProvider
from gs.gaussian_splatting import GaussianSplattingRenderer


class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.max_steps = cfg.max_steps
        self.start = cfg.start

        self.mode = cfg.get("mode", "text_to_3d")

        # Dataset
        if self.mode == "text_to_3d":
            self.dataset = CameraPoseProvider(cfg.data)

        elif self.mode == "image_to_3d":
            self.dataset = SingleViewCameraPoseProvider(cfg.data)

        # Gaussian Splatting Renderer
        initial_values = initialize(cfg.init, image=self.image, depth_map=self.depth_map,
                                    mask=self.mask, c2w=self.dataset.original_out["c2w"],
                                    camera_info=self.dataset.original_out["camera_info"])
        self.renderer = GaussianSplattingRenderer(cfg.renderer, initial_values=initial_values)

    def update(self, step):
        self.dataset.update(step)  # Depends on mode either - "text_to_3d" or "image_to_3d"
        self.renderer.update(step)
        self.guidance.update(step)
        self.prompt_processor.update(step)

    def train_loop(self):
        self.train()

        with tqdm(total=self.max_steps - self.start) as pbar:
            for s in range(self.start, self.max_steps):
                self.step = s
                self.update(self.step)  # To update - dataset, renderer, guidance, and prompt processor
