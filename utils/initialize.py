def point_e_intialize(cfg):
    from utils.point_e_helper import point_e_generate_pcd_from_text


def initialize(cfg, **kwargs):
    if cfg.type == "point_e":
        return point_e_intialize(cfg)
    else:
        raise NotImplementedError(f"{cfg.type} is not implemented")
