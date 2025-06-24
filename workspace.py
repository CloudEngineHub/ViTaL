from collections import defaultdict
import os
import pickle
import ipdb

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from absl import logging

logging.set_verbosity(logging.ERROR)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # hide from other processes using torch
import random
import jax
import time
import orbax.checkpoint
from flax.training import orbax_utils

import numpy as np
from logger import Logger
from pathlib import Path
import utils
import csv
import hydra
import time
from utils import ActionType, Every, Timer, Until
from video import VideoRecorder, TrainVideoRecorder
import matplotlib.pyplot as plt
import json
import xml.etree.ElementTree as ET
import ipdb
import re
from io import BytesIO
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from flax.core.frozen_dict import FrozenDict
from reaching import MolmoZMQClient

act_type = {
    "adroit": ActionType.CONTINUOUS,
    "babyai": ActionType.DISCRETE,
    "dmc": ActionType.CONTINUOUS,
    "metaworld": ActionType.CONTINUOUS,
    "xarm_env": ActionType.CONTINUOUS,
}


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
        print(f"workspace: {self.work_dir}")
        self.cfg = cfg

        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        self.obs_keys = list(self.cfg.suite.pixel_keys + self.cfg.suite.aux_keys)
        self.action_keys = ["action"]

        self.logger = Logger(
            self.work_dir,
            use_tb=self.cfg.use_tb,
            use_wandb=self.cfg.use_wandb,
            mode="bc",
        )

        self.env = hydra.utils.call(
            self.cfg.suite.task_make_fn, expt_type=self.cfg.expt.name
        )

        if self.cfg.temporal_agg:
            self.all_time_actions = np.zeros(
                [
                    # self.max_episode_len ,
                    # self.max_episode_len + self.num_queries,
                    40010,
                    40010 + self.cfg.num_queries,
                    7,  # self.env.action_spec().shape[0],
                ]
            )
        self.agent = hydra.utils.call(
            cfg.expt.agent,
            obs_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
            seed=self.cfg.seed,
            _recursive_=False,
        )

        BaseDatasetClass = hydra.utils.get_class(
            self.cfg.expt.replay_buffer.base_dataset
        )
        self.expert_replay_loader = hydra.utils.call(
            self.cfg.expt.replay_buffer,
            BaseDataset=BaseDatasetClass,
            obs_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
        )

        if "bcrl" in repr(self.agent):
            self.bcrl_setup()

        self.timer = Timer()
        self._global_step = 0
        self._global_episode = 0

        if self.cfg.molmo_reaching:
            # camera intrinstics
            self.K = np.array(
                [
                    [608.456665039062, 0.0, 315.040710449219],
                    [0.0, 608.476440429688, 252.703979492188],
                    [0.0, 0.0, 1.0],
                ]
            )
            # extrinstics
            ext = np.array(
                [
                    [9.03374086e-01, 4.28436641e-01, -1.89025436e-02, -2.67563372e02],
                    [1.84179627e-01, -4.27399950e-01, -8.85102902e-01, 4.63698448e02],
                    [-3.87289460e-01, 7.96097562e-01, -4.65011339e-01, 1.36442562e03],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ]
            )
            self.ext_inv = np.linalg.inv(ext)  # camera→world

        # Setup checkpointer
        self.orbax_ckptr = orbax.checkpoint.PyTreeCheckpointer()

    def bcrl_setup(self):
        self.action_keys = ["action", "action_base"]
        self.sinkhorn_rew_scale_dict = defaultdict(lambda: 200.0)

        self.logger_rl = Logger(
            self.work_dir,
            use_tb=self.cfg.use_tb,
            use_wandb=self.cfg.use_wandb,
            mode="rl",
        )
        # Load and update rl stats
        snapshot_path = Path(self.cfg.expt.agent.bc_snapshot_path)
        # Get parent directory of snapshot_path
        snapshot_dir = snapshot_path.parent
        stats_path = snapshot_dir / "stats.pkl"
        with open(stats_path, "rb") as f:
            stats_bc = pickle.load(f)
        self.expert_replay_loader.update_stats(stats_bc)

        self.train_video_recorder_52 = TrainVideoRecorder(self.work_dir)

        self.reward_type = {}
        self.use_raw = {}
        for key in self.cfg.suite.irl_keys:
            if "pixel" in key:
                self.reward_type[key] = "sinkhorn_cosine"
                self.use_raw[key] = False
            elif "proprio" in key:
                self.reward_type[key] = "sinkhorn_manhattan"
                self.use_raw[key] = True
            else:
                self.reward_type[key] = "sinkhorn_euclidean"
                self.use_raw[key] = True

        if self.cfg.expt.irl:
            data_path = self.cfg.expt.pkg_path
            with open(data_path, "rb") as f:
                data = pickle.load(f)

            for entry in data["observations"]:
                sensor_states = entry["sensor_states"]
                median_baseline = np.median(np.array(sensor_states)[:5], axis=0)
                sensor_states -= median_baseline
                num_sensors = sensor_states.shape[1] // 15
                entry.update(
                    {
                        f"sensor{k}": sensor_states[..., 15 * k : 15 * (k + 1)]
                        for k in range(num_sensors)
                    }
                )

            self.expert_demo = data["observations"]

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def eval(self):
        if "bcrl" in repr(self.agent):
            self.eval_bcrl()
        elif repr(self.agent) == "bc":
            self.eval_bc()
        else:
            raise NotImplementedError(f"Agent {repr(self.agent)} not implemented")

    def train(self):
        if "bcrl" in repr(self.agent):
            self.train_bcrl()
        elif repr(self.agent) == "bc":
            self.train_bc()
        else:
            raise NotImplementedError(f"Agent {repr(self.agent)} not implemented")

    def molmo_reach(self, time_step):
        rgb = time_step.observation["pixels1"][..., ::-1]  # BGR→RGB
        depth = time_step.observation["depth1"]

        prompt = (
            "Mark a point on the white USB port. It's a little white box. "
            'Output only one (x,y) relative coordinate in JSON format, e.g. {"x":0.123,"y":0.456}.'
        )

        # prompt = (
        #     "Mark a point on the red/black switch on the white socket set. "
        #     'Output only one (x,y) relative coordinate in JSON format, e.g. {"x":0.123,"y":0.456}.'
        # )
        # prompt = (
        #     "Mark a point on a grey swiper machine. It's a grey rectangle box with lines connected"
        #     'Output only one (x,y) relative coordinate in JSON format, e.g. {"x":0.123,"y":0.456}.'
        # )

        # prompt = (
        #     "Mark a point on a silver round lock."
        #     'Output only one (x,y) relative coordinate in JSON format, e.g. {"x":0.123,"y":0.456}.'
        # )

        response, (L, T, CW, CH) = self.molmo_c.send_cropped_prompt(prompt, rgb)

        mx = re.search(r"x[^0-9\-\.]*([0-9]+(?:\.[0-9]+)?)", response)
        my = re.search(r"y[^0-9\-\.]*([0-9]+(?:\.[0-9]+)?)", response)
        if mx and my:
            fx, fy = float(mx.group(1)), float(my.group(1))
        else:
            coords = json.loads(response)
            fx, fy = coords["x"], coords["y"]

        x_full = L + utils.to_pixel(fx, CW)
        y_full = T + utils.to_pixel(fy, CH)

        # Sanity-check: save marked full-frame
        img = Image.fromarray(rgb, mode="RGB")
        d = ImageDraw.Draw(img)
        r = 4
        d.ellipse(
            [(x_full - r, y_full - r), (x_full + r, y_full + r)],
            outline=(255, 0, 0),
            width=3,
        )
        d.text(
            (x_full + r + 2, y_full - r),
            f"({x_full},{y_full})",
            fill=(255, 0, 0),
            font=ImageFont.load_default(),
        )

        Z = float(depth[y_full, x_full])
        xyz_cam = utils.pixel_to_camera_frame((x_full, y_full), Z, self.K)

        homog = np.append(xyz_cam, 1.0)
        xyz_wld = (self.ext_inv @ homog)[:3]

        # offsets to align reaching position with bc+rl training domain
        print("Our task is:", self.cfg.task_name)
        if "card" in self.cfg.task_name:
            xyz_wld[0] += 120
            xyz_wld[2] -= 45
        elif "plug" in self.cfg.task_name:
            xyz_wld[0] += 20
            xyz_wld[1] -= 0
            xyz_wld[2] += 0
        elif "usb" in self.cfg.task_name:
            xyz_wld[0] += -60
            xyz_wld[1] += -50
            xyz_wld[2] += 0
        elif "key" in self.cfg.task_name:
            xyz_wld[0] += 30
            xyz_wld[1] -= 0
            xyz_wld[2] += 25

        return xyz_cam, xyz_wld

    def eval_bc(self):
        eval_until_episode = Until(
            self.cfg.suite.num_eval_episodes,
        )
        self._global_episode = 0

        self.train_video_recorder_52 = TrainVideoRecorder(self.work_dir)
        while eval_until_episode(self._global_episode):
            total_reward = 0
            step = 0
            time_step = self.env.reset()
            input("Press to continue")
            time_step = self.env.reset()

            self.train_video_recorder_52.init(
                time_step.observation["pixels52"], enabled=True
            )
            sensor0_data = []

            time.sleep(2)

            if self.cfg.molmo_reaching:
                while True:
                    xyz_cam, xyz_wld = self.molmo_reach(time_step)
                    print("Computed world frame for new start position:", xyz_wld)
                    current_xyz = time_step.observation["proprioceptive"][:3]
                    print("Current position:", current_xyz)
                    error = xyz_wld - current_xyz
                    err_norm = np.linalg.norm(error)
                    user_val = input("Enter s for moving")
                    if user_val == "s":
                        break
                    else:
                        time_step = self.env.reset()

                while err_norm > 1:
                    current_xyz = time_step.observation["proprioceptive"][:3]
                    print("Current position:", current_xyz)
                    Kp = 0.1

                    error = xyz_wld - current_xyz
                    err_norm = np.linalg.norm(error)
                    if err_norm < 1:
                        print(f"Converged: error {err_norm:.4f} < {0.1}")
                        break

                    action_base = np.zeros(7)
                    action_base[:3] = Kp * error

                    max_step = 3
                    step_norm = np.linalg.norm(action_base[:3])
                    if step_norm > max_step:
                        action_base[:3] = action_base[:3] / step_norm * max_step

                    time_step = self.env.step(action_base)
                    self.train_video_recorder_52.record(
                        time_step.observation["pixels52"]
                    )
                    sensor0_data.append(time_step.observation["sensor0"].tolist())

            while not time_step.last():
                if self.cfg.crop_view:
                    time_step.observation["pixels52"] = utils.apply_crop_view(
                        time_step.observation["pixels52"]
                    )

                processed_obs = {
                    k: self.expert_replay_loader.preprocess(time_step.observation[k], k)
                    for k in self.obs_keys
                }
                action = self.agent.eval(processed_obs)

                if self.cfg.temporal_agg:
                    action = action.reshape(
                        -1, self.cfg.num_queries, self.env.action_spec().shape[0]
                    )
                    self.all_time_actions[
                        [step], step : step + self.cfg.num_queries
                    ] = action
                    actions_for_current_step = self.all_time_actions[:, step]
                    actions_populated = np.all(actions_for_current_step != 0, axis=-1)
                    actions_for_current_step = actions_for_current_step[
                        actions_populated
                    ]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_current_step)))
                    exp_weights /= np.sum(exp_weights)
                    action = np.sum(
                        actions_for_current_step * exp_weights[:, None], axis=0
                    )
                action = self.expert_replay_loader.postprocess(action, "action")

                time_step = self.env.step(np.array(action).squeeze())
                self.train_video_recorder_52.record(time_step.observation["pixels52"])
                sensor0_data.append(time_step.observation["sensor0"].tolist())

                total_reward += time_step.reward
                step += 1

            self._global_episode += 1

    def eval_bcrl(self):
        eval_until_episode = Until(
            self.cfg.suite.num_eval_episodes,
        )

        episode_step = 0

        time_step = self.env.reset()
        if self.cfg.crop_view:
            time_step.observation["pixels52"] = utils.apply_crop_view(
                time_step.observation["pixels52"]
            )

        norms = list()

        print("Starting JAX RL Evaling...")

        self.train_video_recorder_52.init(
            time_step.observation["pixels52"], enabled=True
        )

        while eval_until_episode(self._global_episode):
            if self._global_episode == 0 or time_step.last():
                utils.plot_norms(norms, self.work_dir)

                self._global_episode += 1
                self.env.reset()
                input("Press to continue")

                self.train_video_recorder_52.save(
                    f"{self._global_episode}_pixels52.mp4"
                )

                # reset env
                time_step = self.env.reset()
                if self.cfg.molmo_reaching:
                    while True:
                        xyz_cam, xyz_wld = self.molmo_reach(time_step)

                        print("Computed world‐frame for new start position:", xyz_wld)
                        current_xyz = time_step.observation["proprioceptive"][:3]
                        print("Current position:", current_xyz)
                        error = xyz_wld - current_xyz
                        err_norm = np.linalg.norm(error)
                        user_val = input("Enter s for moving")
                        if user_val == "s":
                            break
                        else:
                            time_step = self.env.reset()

                    while err_norm > 1:
                        current_xyz = time_step.observation["proprioceptive"][:3]
                        print("Current position:", current_xyz)
                        Kp = 0.1

                        error = xyz_wld - current_xyz
                        err_norm = np.linalg.norm(error)
                        if err_norm < 1:
                            print(f"Converged: error {err_norm:.4f} < {0.1}")
                            break

                        action_base = np.zeros(7)
                        action_base[:3] = Kp * error

                        max_step = 3
                        step_norm = np.linalg.norm(action_base[:3])
                        if step_norm > max_step:
                            action_base[:3] = action_base[:3] / step_norm * max_step

                        time_step = self.env.step(np.zeros(3), action_base)

                if self.cfg.crop_view:
                    time_step.observation["pixels52"] = utils.apply_crop_view(
                        time_step.observation["pixels52"]
                    )

                episode_step = 1

            features = self.agent.extract_features(
                time_step.observation,
                tuple(self.obs_keys),
                self.expert_replay_loader.stats,
            )

            action_base = self.agent.act_base(features, 0.01)

            if self.cfg.temporal_agg:
                action_base = action_base.reshape(
                    -1,
                    self.cfg.num_queries,
                    7,  # self.env.action_spec().shape[0]
                )
                self.all_time_actions[
                    [self.global_step],
                    self.global_step : self.global_step + self.cfg.num_queries,
                ] = action_base
                actions_for_current_step = self.all_time_actions[:, self.global_step]
                actions_populated = np.all(actions_for_current_step != 0, axis=-1)
                actions_for_current_step = actions_for_current_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_current_step)))
                exp_weights /= np.sum(exp_weights)
                action_base = np.sum(
                    actions_for_current_step * exp_weights[:, None], axis=0
                )
                
            action_base = np.expand_dims(action_base, axis=0)

            if repr(self.agent) == "bcrl_sac":
                temperature = self.agent.temperature.apply_fn(
                    {"params": self.agent.temperature.params}
                )
                rng, key = jax.random.split(self.agent.rng)
                action = self.agent.act_offset(
                    features, action_base, temperature, training=False, rng=key
                )
                self.agent = self.agent.replace(rng=rng)
            else:
                stddev = self.agent.stddev_fn(
                    self.global_step,
                )
                rng, key = jax.random.split(self.agent.rng)
                action = self.agent.act_offset(
                    features, action_base, stddev, training=False, rng=key
                )
                self.agent = self.agent.replace(rng=rng)

            action_base = self.agent.post_process(
                action_base, self.expert_replay_loader.stats, "action_base"
            )

            action = self.agent.post_process(
                action, self.expert_replay_loader.stats, "action"
            )
            print("Offset Action:", action)

            time_step = self.env.step(
                np.array(action).squeeze(), np.array(action_base).squeeze()
            )

            if self.cfg.crop_view:
                time_step.observation["pixels52"] = utils.apply_crop_view(
                    time_step.observation["pixels52"]
                )

            self.train_video_recorder_52.record(time_step.observation["pixels52"])

            episode_step += 1
            self._global_step += 1

        print("Completed JAX RL Evaling..")

    def train_bcrl(self):
        print("Starting JAX RL training...")

        reward_csv_path = self.work_dir / "reward_sums.csv"
        if not reward_csv_path.exists():
            with open(reward_csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Reward Sum", "Reward Mean"])

        train_until_step = Until(
            self.cfg.suite.num_train_steps,
            1,  # self.cfg.suite.action_repeat
        )
        log_every_step = Every(
            self.cfg.suite.log_every_steps,
            1,  # self.cfg.suite.action_repeat
        )
        save_every_step = Every(
            self.cfg.suite.save_every_steps,
            1,  # self.cfg.suite.action_repeat
        )
        seed_until_step = Until(
            self.cfg.suite.num_seed_frames, self.cfg.suite.action_repeat
        )

        episode_step, episode_reward = 0, 0

        time_steps = list()
        actions = list()
        action_bases = list()
        observations = defaultdict(list)

        time_step = self.env.reset()
        if self.cfg.crop_view:
            time_step.observation["pixels52"] = utils.apply_crop_view(
                time_step.observation["pixels52"]
            )
        irl_rewards_history_sum = defaultdict(list)
        episode_outcomes = list()
        norms = list()

        print("Start training")

        self.train_video_recorder_52.init(
            time_step.observation["pixels52"], enabled=True
        )

        metrics = None
        discard_episode = False

        while train_until_step(self.global_step):
            if time_step.last():
                utils.plot_norms(norms, self.work_dir)
                self.env.reset()
                self._global_episode += 1
                self.env.reset()

                success = time_step.observation["goal_achieved"]
                if self.cfg.human_in_loop:
                    try:
                        user_val = float(
                            input(
                                f"Episode {self._global_episode} finished. "
                                "Enter additional reward to assign at the last timestep (or 0): "
                            )
                        )

                    except ValueError:
                        print("  → invalid input, using 0")
                        user_val = 0.0
                    if user_val > 0:
                        print("Succesful")
                        success = True
                    else:
                        print("Failure")
                        success = False

                episode_outcomes.append(success)

                # wait until all the metrics schema is populated
                observations = jax.tree.map(lambda x: np.stack(x, 0), observations)
                if self.cfg.expt.irl:
                    new_rewards_sum = 0
                    new_rewards_mean = 0
                    new_rewards = np.zeros(
                        episode_step,
                    )
                    if self.cfg.expt.min_reward:
                        min_rewards = np.full(episode_step, np.inf, dtype=float)
                    per_key_episode_rewards = {}

                    for key in self.cfg.suite.irl_keys:
                        key_rewards = self.agent.multi_rewarder(
                            observations[key],
                            self.expert_demo,
                            key,
                            self.expert_replay_loader.stats,
                            self.sinkhorn_rew_scale_dict,
                            self.reward_type[key],
                            self.use_raw[key],
                        )

                        if np.mean(key_rewards) > -1e-3:
                            discard_episode = True
                            print(
                                "\033[91mWarning: Invalid Reward. Discard Episode.\033[0m"
                            )

                        if self.cfg.human_in_loop:
                            if user_val > 0:
                                key_rewards[-1] += 5.0
                            else:
                                key_rewards[-1] -= 5.0

                            # key_rewards += user_val

                        if self.cfg.expt.min_reward:
                            print(
                                "\033[91mWe are taking the min among different modality rewards.\033[0m"
                            )
                            min_rewards = np.minimum(key_rewards, min_rewards)
                        new_rewards += key_rewards
                        new_rewards_sum += np.sum(key_rewards)
                        new_rewards_mean += np.mean(key_rewards)

                        per_key_episode_rewards[key] = np.sum(key_rewards)
                        irl_rewards_history_sum[key].append(
                            per_key_episode_rewards[key]
                        )

                        key_rewards_arr = np.array(key_rewards)
                        timesteps = np.arange(key_rewards_arr.shape[0])

                        if not discard_episode:
                            utils.plot_reward_vs_timestep(
                                key,
                                timesteps,
                                key_rewards,
                                self._global_episode,
                                self.work_dir,
                                normalize=False,
                            )

                    print("IRL Reward:", new_rewards_sum)
                    if not discard_episode:
                        if (
                            self._global_episode
                            > self.cfg.expt.num_scale_estimation_episodes
                        ):
                            with open(reward_csv_path, mode="a", newline="") as file:
                                writer = csv.writer(file)
                                writer.writerow(
                                    [
                                        self._global_episode,
                                        new_rewards_sum,
                                        new_rewards_mean,
                                    ]
                                )

                    if self.cfg.expt.auto_rew_scale and not self.cfg.expt.load_rl:
                        # After several episodes, compute the average reward for each key from the first 5 episodes.
                        if (
                            self._global_episode
                            == self.cfg.expt.num_scale_estimation_episodes
                        ):
                            for key in list(self.cfg.suite.irl_keys):
                                firstn_rewards = irl_rewards_history_sum.get(key, [])[
                                    : self.cfg.expt.num_scale_estimation_episodes
                                ]
                                if (
                                    firstn_rewards
                                    and np.abs(np.mean(firstn_rewards)) > 0
                                ):
                                    # Instead of mean,for each episode, compute its range (= max–min)
                                    # ranges = [float(np.max(r) - np.min(r)) for r in firstn_rewards]
                                    # mean_range = float(np.mean(ranges))

                                    mean_range = np.mean(firstn_rewards)

                                    self.sinkhorn_rew_scale_dict[key] = (
                                        self.cfg.expt.sinkhorn_rew_scale
                                        * self.cfg.expt.auto_rew_scale_factor
                                        / float(np.abs(mean_range))
                                    )
                                    print(
                                        f"Set sinkhorn_rew_scale_dict[{key}] based on average reward {mean_range} from first {self.cfg.expt.num_scale_estimation_episodes} episodes"
                                    )

                if not discard_episode and self._global_episode >= (
                    self.cfg.expt.auto_rew_scale
                    * self.cfg.expt.num_scale_estimation_episodes
                ):
                    for i, elt in enumerate(time_steps):
                        if i > 2:
                            ####################################################################
                            elt = elt._replace(reward=time_steps[i].reward / 4.0)
                            ####################################################################
                            if self.cfg.expt.irl:
                                if self.cfg.expt.min_reward:
                                    elt = elt._replace(reward=min_rewards[i])
                                else:
                                    elt = elt._replace(reward=new_rewards[i])
                            self.expert_replay_loader.insert_from_timestep(
                                elt, bg_augs=self.cfg.bg_augs, preprocess=True
                            )

                print("Task Successful: ", success)

                self.train_video_recorder_52.save(
                    f"{self._global_episode}_pixels52.mp4"
                )

                # reset env
                time_steps = list()
                actions = list()
                action_bases = list()
                observations = defaultdict(list)
                discard_episode = False

                time_step = self.env.reset()
                if self.cfg.molmo_reaching:
                    while True:
                        xyz_cam, xyz_wld = self.molmo_reach(time_step)

                        print("Computed world‐frame for new start position:", xyz_wld)
                        current_xyz = time_step.observation["proprioceptive"][:3]
                        print("Current position:", current_xyz)
                        error = xyz_wld - current_xyz
                        err_norm = np.linalg.norm(error)
                        user_val = input("Enter s for moving")
                        if user_val == "s":
                            break
                        else:
                            time_step = self.env.reset()

                    while err_norm > 1:
                        current_xyz = time_step.observation["proprioceptive"][:3]
                        print("Current position:", current_xyz)
                        Kp = 0.1

                        error = xyz_wld - current_xyz
                        err_norm = np.linalg.norm(error)
                        if err_norm < 1:
                            print(f"Converged: error {err_norm:.4f} < {0.1}")
                            break

                        action_base = np.zeros(7)
                        action_base[:3] = Kp * error

                        max_step = 3
                        step_norm = np.linalg.norm(action_base[:3])
                        if step_norm > max_step:
                            action_base[:3] = action_base[:3] / step_norm * max_step

                        time_step = self.env.step(np.zeros(3), action_base)

                if self.cfg.crop_view:
                    time_step.observation["pixels52"] = utils.apply_crop_view(
                        time_step.observation["pixels52"]
                    )

                time_steps.append(time_step)
                actions.append(time_step.action)
                action_bases.append(time_step.action_base)
                for key in self.cfg.suite.irl_keys:
                    observations[key].append(time_step.observation[key])

                self.train_video_recorder_52.init(
                    time_step.observation["pixels52"], enabled=True
                )

                episode_step = 1
                episode_reward = 0

            features = self.agent.extract_features(
                time_step.observation,
                tuple(self.obs_keys),
                self.expert_replay_loader.stats,
            )

            action_base = self.agent.act_base(features, stddev=0.01)

            if self.cfg.temporal_agg:
                action_base = action_base.reshape(
                    -1,
                    self.cfg.num_queries,
                    7,  # self.env.action_spec().shape[0]
                )
                self.all_time_actions[
                    [self.global_step],
                    self.global_step : self.global_step + self.cfg.num_queries,
                ] = action_base
                actions_for_current_step = self.all_time_actions[:, self.global_step]
                actions_populated = np.all(actions_for_current_step != 0, axis=-1)
                actions_for_current_step = actions_for_current_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_current_step)))
                exp_weights /= np.sum(exp_weights)
                action_base = np.sum(
                    actions_for_current_step * exp_weights[:, None], axis=0
                )

            action_base = np.expand_dims(action_base, axis=0)

            if repr(self.agent) == "bcrl_sac":
                temperature = self.agent.temperature.apply_fn(
                    {"params": self.agent.temperature.params}
                )
                rng, key = jax.random.split(self.agent.rng)
                action = self.agent.act_offset(
                    features, action_base, temperature, training=True, rng=key
                )
                self.agent = self.agent.replace(rng=rng)
            else:
                stddev = self.agent.stddev_fn(
                    self.global_step,
                )
                rng, key = jax.random.split(self.agent.rng)
                action = self.agent.act_offset(
                    features, action_base, stddev, training=True, rng=key
                )
                self.agent = self.agent.replace(rng=rng)

            action_base = self.agent.post_process(
                action_base, self.expert_replay_loader.stats, "action_base"
            )

            action = self.agent.post_process(
                action, self.expert_replay_loader.stats, "action"
            )
            print(f"Offset Action: {action}")

            if save_every_step(self.global_step):
                self.save_snapshot()

            # try to update the agent
            if not seed_until_step(self.global_step):
                # Update
                self.agent, batch_rng = self.agent.split_rng(1)

                batch = self.expert_replay_loader.sample(
                    batch_size=self.cfg.batch_size,
                    rng=batch_rng,
                    bg_augs=self.cfg.bg_augs,
                )

                self.agent, metrics = self.agent.update(
                    batch, self.global_step, utd_ratio=self.cfg.expt.utd_ratio
                )
                if log_every_step(self.global_step):
                    with self.logger_rl.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        elapsed_time, total_time = self.timer.reset()
                        episode_frame = episode_step * self.cfg.suite.action_repeat
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("buffer_size", len(self.expert_replay_loader))
                        log("step", self.global_step)

                        for name, value in metrics.items():
                            log(name, value)

            time_step = self.env.step(
                np.array(action).squeeze(), np.array(action_base).squeeze()
            )
            episode_reward += time_step.reward
            if self.cfg.crop_view:
                time_step.observation["pixels52"] = utils.apply_crop_view(
                    time_step.observation["pixels52"]
                )

            sensor_reading = time_step.observation["sensor0"]
            norm_value = np.linalg.norm(sensor_reading).item()

            norms.append(norm_value)

            time_steps.append(time_step)
            actions.append(time_step.action)
            action_bases.append(time_step.action_base)
            for key in self.cfg.suite.irl_keys:
                observations[key].append(time_step.observation[key])

            self.train_video_recorder_52.record(time_step.observation["pixels52"])

            episode_step += 1
            self._global_step += 1

        print("Completed JAX RL training..")

    def train_bc(self):
        train_until_step = Until(
            self.cfg.suite.num_train_steps,
            1,  # self.cfg.suite.action_repeat
        )
        log_every_step = Every(
            self.cfg.suite.log_every_steps,
            1,  # self.cfg.suite.action_repeat
        )
        save_every_step = Every(
            self.cfg.suite.save_every_steps,
            1,  # self.cfg.suite.action_repeat
        )

        print("Start JAX BC training")

        while train_until_step(self.global_step):
            # Save first so that restored checkpoint is aligned with the previous
            # training loop
            if save_every_step(self.global_step):
                self.save_snapshot()
            self.agent, batch_rng = self.agent.split_rng(1)

            batch = self.expert_replay_loader.sample(
                batch_size=self.cfg.batch_size,
                rng=batch_rng,
                keys=self.obs_keys + self.action_keys,
            )
            self.agent, metrics = self.agent.update(batch, self.global_step)
            self.logger.log_metrics(metrics, self.global_frame, ty="train")

            if log_every_step(self.global_step):
                _, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("actor_loss", metrics["actor_loss"])
                    log("step", self.global_step)

            self._global_step += 1

        print("Completed JAX BC training..")

    def save_snapshot(self):
        snapshot_dir = self.work_dir / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot = snapshot_dir / f"{self.global_step}"
        agent_state = {
            "agent": self.agent.get_save_state(),
            "step": self.global_step,
        }
        stats_path = snapshot_dir / "stats.pkl"
        if not os.path.exists(stats_path):
            with open(stats_path, "wb") as f:
                pickle.dump(self.expert_replay_loader.stats, f)

        self.orbax_ckptr.save(
            snapshot,
            agent_state,
            orbax_utils.save_args_from_target(agent_state),
            force=True,
        )

    def load_snapshot(self, snapshot):
        target = {
            "agent": self.agent.get_save_state(),
            "step": self.global_step,
        }
        restored_agent_state = self.orbax_ckptr.restore(snapshot, item=target)
        self.agent = self.agent.load_state(restored_agent_state["agent"])
