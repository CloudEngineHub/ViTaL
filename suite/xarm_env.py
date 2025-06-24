from collections import deque
from typing import Any, NamedTuple

import gym
from gym import Wrapper, spaces

# from gym.wrappers import FrameStack

import xarm_env
import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep

import cv2
import time
# from libero.libero import benchmark, get_libero_path
# from libero.libero.envs import OffScreenRenderEnv


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(
        self,
        env,
        pixel_keys=["pixels0"],
        aux_keys=["proprioceptive"],
    ):
        self._env = env
        self.pixel_keys = pixel_keys
        self.aux_keys = aux_keys

        # TODO: Remove environment reset requirement
        obs = self._env.reset()

        pixels = obs[pixel_keys[0]]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=pixels.shape, dtype=pixels.dtype
        )

        # Action spec
        action_spec = self._env.action_space
        # self._action_spec = specs.BoundedArray(
        #     action_spec[0].shape, np.float32, action_spec[0], action_spec[1], "action"
        # )
        self._action_spec = specs.Array(
            shape=action_spec.shape, dtype=action_spec.dtype, name="action"
        )
        # Observation spec
        # robot_state = np.concatenate(
        #     [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        # )
        self._obs_spec = {}
        for key in pixel_keys:
            self._obs_spec[key] = specs.BoundedArray(
                shape=obs[key].shape,
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name=key,
            )

        self._obs_spec["proprioceptive"] = specs.BoundedArray(
            shape=obs["proprioceptive"].shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="proprioceptive",
        )

        for key in aux_keys:
            if key.startswith("sensor"):
                self._obs_spec[key] = specs.BoundedArray(
                    shape=obs[key].shape,
                    dtype=np.float32,
                    minimum=-np.inf,
                    maximum=np.inf,
                    name=key,
                )

        self.render_image = None

    def reset(self, **kwargs):
        self._step = 0
        obs = self._env.reset(**kwargs)

        observation = {}
        for key in self.pixel_keys:
            observation[key] = obs[key]
        observation["proprioceptive"] = obs["proprioceptive"]
        # if "sensor" in self.aux_keys:
        #     observation["sensor"] = obs["sensor"]
        for key in self.aux_keys:
            if key.startswith("sensor"):
                observation[key] = obs[key]

        for key in obs.keys():
            if key.startswith("depth") or key == "pixels1":
                observation[key] = obs[key]

        observation["goal_achieved"] = False
        # import ipdb;ipdb.set_trace()
        return observation

    def step(self, action):
        self._step += 1
        obs, reward, truncated, terminated, info = self._env.step(action)
        done = truncated or terminated
        # self.render_image = obs["agentview_image"][::-1, :]

        observation = {}
        for key in self.pixel_keys:
            observation[key] = obs[key]
        observation["proprioceptive"] = obs["proprioceptive"]
        if "sensor" in self.aux_keys:
            observation["sensor"] = obs["sensor"]
        for key in self.aux_keys:
            if key.startswith("sensor"):
                observation[key] = obs[key]
        # observation["pixels"] = obs["agentview_image"][::-1, :]
        # observation["pixels_egocentric"] = obs["robot0_eye_in_hand_image"][::-1, :]
        # observation["proprioceptive"] = np.concatenate(
        #     [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        # )
        # # get state
        # state = self._env.get_sim_state()  # TODO: Change to robot state
        # observation["features"][: state.shape[0]] = state
        observation["goal_achieved"] = done  # (self._step == self._max_episode_len)
        return observation, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        # return cv2.resize(self.render_image, (width, height))
        return cv2.resize(self._env.render(mode), (width, height))

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 0.99
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames

        self.pixel_keys = [
            keys for keys in env.observation_spec().keys() if "pixels" in keys
        ]
        wrapped_obs_spec = env.observation_spec()[self.pixel_keys[0]]

        # frames lists
        self._frames = {}
        for key in self.pixel_keys:
            self._frames[key] = deque([], maxlen=num_frames)
        # self._frames = deque([], maxlen=num_frames)
        # self._frames_egocentric = deque([], maxlen=num_frames)

        pixels_shape = wrapped_obs_spec.shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = {}
        self._obs_spec["proprioceptive"] = self._env.observation_spec()[
            "proprioceptive"
        ]
        for key in self._env.observation_spec().keys():
            if key.startswith("sensor"):
                self._obs_spec[key] = self._env.observation_spec()[key]
            if key.startswith("digit"):
                self._obs_spec[key] = self._env.observation_spec()[key]
        # if "sensor" in self._env.observation_spec().keys():
        #     self._obs_spec["sensor"] = self._env.observation_spec()["sensor"]
        for key in self.pixel_keys:
            self._obs_spec[key] = specs.BoundedArray(
                shape=np.concatenate(
                    [pixels_shape[:2], [pixels_shape[2] * num_frames]], axis=0
                ),
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name=key,
            )
        # self._obs_spec["pixels"] = specs.BoundedArray(
        #     shape=np.concatenate(
        #         [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
        #     ),
        #     dtype=np.uint8,
        #     minimum=0,
        #     maximum=255,
        #     name="pixels",
        # )
        # self._obs_spec["pixels_egocentric"] = specs.BoundedArray(
        #     shape=np.concatenate(
        #         [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
        #     ),
        #     dtype=np.uint8,
        #     minimum=0,
        #     maximum=255,
        #     name="pixels_egocentric",
        # )

    def _transform_observation(self, time_step):
        for key in self.pixel_keys:
            assert len(self._frames[key]) == self._num_frames
        # assert len(self._frames) == self._num_frames
        # assert len(self._frames_egocentric) == self._num_frames
        obs = {}
        for key in self.pixel_keys:
            obs[key] = np.concatenate(list(self._frames[key]), axis=0)
        # obs["pixels"] = np.concatenate(list(self._frames), axis=0)
        # obs["pixels_egocentric"] = np.concatenate(list(self._frames_egocentric), axis=0)
        obs["proprioceptive"] = time_step.observation["proprioceptive"]
        try:
            for key in time_step.observation.keys():
                if key.startswith("sensor"):
                    obs[key] = time_step.observation[key]
                if key.startswith("depth") or key == "pixels1":
                    obs[key] = time_step.observation[key]
            # obs["sensor"] = time_step.observation["sensor"]
        except KeyError:
            pass
        obs["goal_achieved"] = time_step.observation["goal_achieved"]

        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = {}
        for key in self.pixel_keys:
            pixels[key] = time_step.observation[key]
            if len(pixels[key].shape) == 4:
                pixels[key] = pixels[key][0]
            # TODO: Fix this transpose
            # pixels[key] = pixels[key].transpose(2, 0, 1)
        return pixels
        # return [pixels[key].transpose(2, 0, 1).copy() for key in self.pixel_keys]

        # # pixels = time_step.observation["pixels"] ixels_egocentric"]

        # # remove batch dim
        # if len(pixels.shape) == 4:
        #     pixels = pixels[0]
        # if len(pixels_egocentric.shape) == 4:
        #     pixels_egocentric = pixels_egocentric[0]
        # return (
        #     pixels.transpose(2, 0, 1).copy(),
        #     pixels_egocentric.transpose(2, 0, 1).copy(),
        # )

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        # pixels, pixels_egocentric = self._extract_pixels(time_step)
        pixels = self._extract_pixels(time_step)
        for key in self.pixel_keys:
            for _ in range(self._num_frames):
                self._frames[key].append(pixels[key])
        # for _ in range(self._num_frames):
        #     self._frames.append(pixels)
        #     self._frames_egocentric.append(pixels_egocentric)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        # pixels, pixels_egocentric = self._extract_pixels(time_step)
        for key in self.pixel_keys:
            self._frames[key].append(pixels[key])
        # self._frames.append(pixels)
        # self._frames_egocentric.append(pixels_egocentric)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

        # Action spec
        wrapped_action_spec = env.action_spec()
        # self._action_spec = specs.BoundedArray(
        #     wrapped_action_spec.shape,
        #     np.float32,
        #     wrapped_action_spec.minimum,
        #     wrapped_action_spec.maximum,
        #     "action",
        # )
        self._action_spec = specs.Array(
            shape=wrapped_action_spec.shape, dtype=dtype, name="action"
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        # Make time step for action space
        observation, reward, done, info = self._env.step(action)
        # step_type = StepType.LAST if observation['goal_achieved'] else StepType.MID
        step_type = StepType.LAST if done else StepType.MID
        # step_type = (
        #     StepType.LAST
        #     if (
        #         self._env._step == self._env._max_episode_len
        #         or observation["goal_achieved"]
        #     )
        #     else StepType.MID
        # )

        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec
        # return self._env.action_spec()

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return TimeStep(
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepOffset(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    action_base: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self, time_step, observation=None, action=None, reward=None, discount=None
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class OffsetActionWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

        wrapped_action_spec = env.action_spec()
        self._action_spec = {
            "action_base": wrapped_action_spec,
            "action": specs.Array(
                shape=(3,), dtype=wrapped_action_spec.dtype, name="action"
            ),
        }

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action, action_base):
        net_action = self._combine_actions(action, action_base)
        time_step = self._env.step(net_action)
        return self._augment_time_step(time_step, action, action_base)

    def _combine_actions(self, action, action_base):
        # action base comes from bc and action is the offset
        action_base[:3] = action_base[:3] + action
        return action_base

    def _augment_time_step(self, time_step, action=None, action_base=None):
        if action is None:
            action_spec = self.action_spec()["action"]
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        if action_base is None:
            action_base_spec = self.action_spec()["action_base"]
            action_base = np.zeros(action_base_spec.shape, dtype=action_base_spec.dtype)
        return ExtendedTimeStepOffset(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
            action_base=action_base,
        )

    def _replace(
        self,
        time_step,
        observation=None,
        action=None,
        reward=None,
        discount=None,
        action_base=None,
    ):
        ext_time_step = super()._replace(
            time_step, observation, action, reward, discount
        )
        if action_base is None:
            action_base = time_step.action_base
        return ExtendedTimeStepOffset(
            observation=ext_time_step.observation,
            step_type=ext_time_step.step_type,
            action=ext_time_step.action,
            reward=ext_time_step.reward,
            discount=ext_time_step.discount,
            action_base=action_base,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


class SensorSafetyWrapper(dm_env.Environment):
    def __init__(self, env, sensor_key="sensor0", threshold=400.0):
        self._env = env
        self.sensor_key = sensor_key
        self.threshold = threshold

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        time_step = self._env.step(action)
        sensor_value = time_step.observation.get(self.sensor_key, None)

        if sensor_value is not None:
            norm_val = np.linalg.norm(sensor_value)
            if norm_val > self.threshold:
                print(
                    "\033[91mWarning: Anyskin sensor detected unusually high readings. For safety reasons, the XArm will be reset.\033[0m"
                )
                time_step = time_step._replace(step_type=StepType.LAST)
                time_step.observation["goal_achieved"] = False

        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class TaskSafetyWrapper(dm_env.Environment):
    def __init__(self, env, rl_task_name, sensor_key="proprioceptive"):
        self._env = env
        self.rl_task_name = rl_task_name
        self.sensor_key = sensor_key
        self.done = False

    def reset(self, **kwargs):
        self.done = False
        return self._env.reset(**kwargs)

    def step(self, action):
        time_step = self._env.step(action)
        current_state = time_step.observation.get(self.sensor_key, None)

        if current_state is not None and not self.done:
            goal_achieved, reset_env = self._check_task_safety(
                current_state, self.rl_task_name
            )
            if reset_env:
                self.done = True
                time_step = time_step._replace(step_type=StepType.LAST)
                time_step.observation["goal_achieved"] = goal_achieved

        return time_step

    def _check_task_safety(self, current_state, rl_task_name):
        goal_achieved = False
        reset = False

        if rl_task_name == "cup_insertion":
            if current_state[2] < 365:
                print("\033[91mWarning: XArm is too low. Reset now.\033[0m")
                time.sleep(1)
                goal_achieved = True
                reset = True

        elif rl_task_name == "plug_insertion":
            if current_state[2] < 273:
                print("\033[91mWarning: XArm is too low. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = True
                reset = True

            if current_state[2] > 440:
                print("\033[91mWarning: XArm is too high. Reset now.\033[0m")
                time.sleep(1)
                print("Current State:", current_state)
                goal_achieved = False
                reset = True

            if abs(current_state[4]) > 0.05 or abs(current_state[5]) > 0.04:
                print("\033[91mWarning: XArm is in wrong gesture. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = False
                reset = True

        elif rl_task_name == "usb_insertion":
            if current_state[2] < 265:
                print("\033[91mWarning: XArm is too low. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = True
                reset = True

            if current_state[2] > 480:
                print("\033[91mWarning: XArm is too high. Reset now.\033[0m")
                time.sleep(1)
                print("Current State:", current_state)
                goal_achieved = False
                reset = True

            if abs(current_state[4]) > 0.05 or abs(current_state[5]) > 0.04:
                print("\033[91mWarning: XArm is in wrong gesture. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = False
                reset = True
        elif rl_task_name == "picking":
            if current_state[2] < 210:
                print("\033[91mWarning: XArm is too low. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = True
                reset = True

            if current_state[2] > 480:
                print("\033[91mWarning: XArm is too high. Reset now.\033[0m")
                time.sleep(1)
                print("Current State:", current_state)
                goal_achieved = False
                reset = True

            if abs(current_state[4]) > 0.05 or abs(current_state[5]) > 0.04:
                print("\033[91mWarning: XArm is in wrong gesture. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = False
                reset = True

        elif rl_task_name == "key_insertion":
            if current_state[2] < 253:
                print("\033[91mWarning: XArm is too low. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = True
                reset = True

            if current_state[2] > 480:
                print("\033[91mWarning: XArm is too high. Reset now.\033[0m")
                time.sleep(1)
                print("Current State:", current_state)
                goal_achieved = False
                reset = True

            if abs(current_state[4]) > 0.05 or abs(current_state[5]) > 0.04:
                print("\033[91mWarning: XArm is in wrong gesture. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = False
                reset = True

        elif rl_task_name == "key_unlock":
            if current_state[2] < 356:
                print("\033[91mWarning: XArm is too low. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = False
                reset = True

            if current_state[2] > 455:
                print("\033[91mWarning: XArm is too high. Reset now.\033[0m")
                time.sleep(1)
                print("Current State:", current_state)
                goal_achieved = False
                reset = True

            if abs(current_state[4]) > 0.40:
                print("\033[91mWarning: XArm is in wrong gesture. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = False
                reset = True

        elif rl_task_name == "card_swiping":
            if current_state[2] < 235:
                print("\033[91mWarning: XArm is too low. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = False
                reset = True

            if current_state[0] < 285:
                print("\033[91mWarning: XArm is too left. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = True
                reset = True

            if current_state[2] > 460:
                print("\033[91mWarning: XArm is too high. Reset now.\033[0m")
                time.sleep(1)
                print("Current State:", current_state)
                goal_achieved = False
                reset = True

            if abs(current_state[4]) > 0.15 or abs(current_state[5]) > 0.12:
                print("\033[91mWarning: XArm is in wrong gesture. Reset now.\033[0m")
                print("Current State:", current_state)
                time.sleep(1)
                goal_achieved = False
                reset = True
        return goal_achieved, reset

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(
    frame_stack,
    action_repeat,
    seed,
    height,
    width,
    pixel_keys,
    aux_keys,
    use_robot,  # True means use_robot=True
    mask_view,
    molmo_reaching,
    task_name,
    expt_type="bc",
):
    env = gym.make(
        "Robot-v1",
        height=height,
        width=width,
        use_robot=use_robot,
        mask_view=mask_view,
        molmo_reaching=molmo_reaching,
    )
    # env.seed(seed)

    # apply wrappers
    env = RGBArrayAsObservationWrapper(
        env,
        pixel_keys=pixel_keys,
        aux_keys=aux_keys,
    )
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    env = TaskSafetyWrapper(env, rl_task_name=task_name, sensor_key="proprioceptive")
    print("Task in the env is:", task_name)

    env = SensorSafetyWrapper(env, sensor_key="sensor0", threshold=600.0)

    if "bcrl" in expt_type:
        env = OffsetActionWrapper(env)

    return env
