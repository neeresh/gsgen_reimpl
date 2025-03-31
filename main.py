import hydra

from rich.console import Console

from trainer import Trainer

console = Console()


# @hydra.main(version_base="1.3", config_path="conf", config_name="trainer")
@hydra.main(version_base="1.3", config_path="conf")
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train_loop()


if __name__ == '__main__':
    main()
