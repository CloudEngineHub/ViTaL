import hydra

from workspace import Workspace as W


@hydra.main(config_path="cfgs", config_name="config", version_base=None)
def main(cfg):
    workspace = W(cfg)
    if cfg.load_model:
        print(cfg.model_path)
        workspace.load_snapshot(snapshot=cfg.model_path)

    workspace.train()


if __name__ == "__main__":
    main()
