from torch.utils.data import Dataset


def CameraPoseProvider(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def update(self, step):
        self.step = step