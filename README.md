# **Touch begins where vision ends: Generalizable policies for contact-rich manipulation**
[[Paper]](https://arxiv.org/abs/2506.13762) [[Project Website]](https://vitalprecise.github.io/)

[Zifan Zhao](https://www.zifanzhao.com)¹, [Siddhant Haldar](http://siddhanthaldar.github.io/)², [Jinda Cui](https://www.jindacui.com/bio/)³, [Lerrel Pinto](https://www.lerrelpinto.com/)², [*Raunaq Bhirangi](https://raunaqbhirangi.github.io/)²

¹New York University Shanghai, ²New York University, ³Honda Research

*Corresponding author: raunaqbhirangi@nyu.edu

---
This repository provides code for "Touch begins where vision ends: Generalizable policies for contact-rich manipulation". It supports jax-based behaviour cloning and residual RL modules. Semantic augmentation pipelines and VLM-guided reaching extension are also included.

---

## 1. Installation

1. Create a Conda environment:
   ```bash
   conda env create -n vital python=3.10
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```
4. Configure project root:
   - Create `cfgs/local_config.yaml` with:
     ```yaml
     root_dir: /path/to/vital/
     ```
5. Update experiment parameters in `cfgs/*.yaml`, notably `suite/xarm.yaml` for dataset paths.

---

## 2. Usage

- **BC raining:**
  ```bash
  python train.py
  ```
- **BC Evaluation:**
  ```bash
  python eval.py model_path=/path/to/checkpoint
  ```

- **RL raining:**
  ```bash
  python train.py expt=bcrl_sac offset_action_scale=2.0
  ```
- **RL Evaluation:**
  ```bash
  python eval.py expt=bcrl_sac offset_action_scale=2.0 model_path=/path/to/checkpoint
  ```
---

## 3. Semantic Augmentation Pipeline

Preprocessing scripts for domain randomization and augmentation using [DIFT](https://github.com/Tsingularity/dift), [SAM2](https://github.com/facebookresearch/sam2), and [XMem](https://github.com/hkchengrex/XMem). For each module you should create corresponding environment from the original repo.

### 3.1 DIFT

```bash
cd dift
python process_xarm_pipeline.py
```

**Prepare anchor data** for each task:
   ```
   anchor_data/{task_name}/base/
   ├── gripper_mask.png
   ├── object_mask.png
   ├── target_mask.png
   └── dift_feature_map.pt
   ```

### 3.2 SAM2

```bash
cd sam2
python process_xarm_pipeline.py
```

### 3.3 XMem

```bash
cd xmem
python process_xarm_pipeline.py
```

Populate `augmented_backgrounds/` with randomly generated background images for augmentation.

---

## 4. Reinforcement Learning Workflow


1. **Launch preprocessing servers** in separate terminals:
   ```bash
   # Terminal 1
   cd dift && python dift_server.py

   # Terminal 2
   cd sam2 && python sam2_server.py

   # Terminal 3
   cd xmem && ./scripts/download_models.sh && python xmem_server.py
   ```

---

## 5. VLM-Guided Reaching Extension

Enable visuotactile reaching with the [Molmo](https://github.com/allenai/molmo) server:

```bash
cd molmo
python molmo_server.py
```

Set `molmo_reaching: true` in your configuration file to activate the VLM-based reaching module.

---
