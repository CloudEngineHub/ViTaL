import sys

sys.path.append(".")

import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="../cfgs", config_name="config.yaml", version_base=None)
def main(cfg):
    print("\n✅ Resolved Config:\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))  # 🔥 Fully resolves interpolations


if __name__ == "__main__":
    main()
