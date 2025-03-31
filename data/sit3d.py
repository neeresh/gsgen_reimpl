from data import CameraPoseProvider


class SingleViewCameraPoseProvider(CameraPoseProvider):
    def __init__(self, cfg):
        self.cfg = cfg