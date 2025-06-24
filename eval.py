from pathlib import Path
import hydra

from workspace import Workspace as W


@hydra.main(config_path="cfgs", config_name="config", version_base=None)
def main(cfg):
    cfg.eval = True
    workspace = W(cfg)

    snapshot = Path(cfg.model_path)
    if not snapshot.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot}")
    print(f"Loading snapshot from {snapshot}")
    workspace.load_snapshot(snapshot)

    workspace.eval()


if __name__ == "__main__":
    main()
