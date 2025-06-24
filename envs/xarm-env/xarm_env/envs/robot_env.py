import gym
from gym import spaces
import cv2
import numpy as np

# import pybullet
# import pybullet_data
import pickle
import zmq
import torch
import ipdb
from PIL import Image
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from openteach.utils.network import create_request_socket, ZMQCameraSubscriber
from xarm_env.envs.constants import *


def get_quaternion_orientation(cartesian):
    """
    Get quaternion orientation from axis angle representation
    """
    pos = cartesian[:3]
    ori = cartesian[3:]
    r = R.from_rotvec(ori)
    quat = r.as_quat()
    return np.concatenate([pos, quat], axis=-1)


class RobotEnv(gym.Env):
    def __init__(
        self,
        height=224,
        width=224,
        use_robot=True,  # True when robot used
        mask_view=False,
        molmo_reaching=False,
        sensor_type="reskin",
        subtract_sensor_baseline=True,
        use_sensor_diffs=False,
        separate_sensors=True,
    ):
        super(RobotEnv, self).__init__()
        self.height = height
        self.width = width

        self.use_robot = use_robot
        self.sensor_type = sensor_type
        self.digit_keys = ["digit80", "digit81"]

        self.subtract_sensor_baseline = subtract_sensor_baseline
        self.use_sensor_diffs = use_sensor_diffs
        self.separate_sensors = separate_sensors
        self.sensor_prev_state = None
        self.sensor_baseline = None

        self.feature_dim = 8  # 10  # 7
        self.proprio_dim = 8

        self.n_sensors = 2
        self.per_sensor_dim = 15
        self.sensor_dim = (
            self.per_sensor_dim * self.n_sensors
            if not self.separate_sensors
            else self.per_sensor_dim
        )
        self.action_dim = 7

        # Robot limits
        # self.cartesian_delta_limits = np.array([-10, 10])

        self.n_channels = 3
        self.reward = 0

        self.done = False
        self.mask_view = mask_view
        self.molmo_reaching = molmo_reaching

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        if self.use_robot:
            # camera subscribers
            self.image_subscribers = {}
            self.depth_subscribers = {}

            for cam_idx in list(CAM_SERIAL_NUMS.keys()):
                port = CAMERA_PORT_OFFSET + cam_idx
                self.image_subscribers[cam_idx] = ZMQCameraSubscriber(
                    host=HOST_ADDRESS,
                    port=port,
                    topic_type="RGB",
                )

            for cam_idx in list(CAM_SERIAL_NUMS.keys()):
                port = DEPTH_PORT_OFFSET + cam_idx
                self.depth_subscribers[cam_idx] = ZMQCameraSubscriber(
                    host=HOST_ADDRESS,
                    port=port,
                    topic_type="Depth",
                )

            # for fish_eye_cam_idx in range(len(FISH_EYE_CAM_SERIAL_NUMS)):
            for fish_eye_cam_idx in list(FISH_EYE_CAM_SERIAL_NUMS.keys()):
                port = FISH_EYE_CAMERA_PORT_OFFSET + fish_eye_cam_idx
                self.image_subscribers[fish_eye_cam_idx] = ZMQCameraSubscriber(
                    host=HOST_ADDRESS,
                    port=port,
                    topic_type="RGB",
                )

            # action request port
            self.action_request_socket = create_request_socket(
                HOST_ADDRESS, DEPLOYMENT_PORT
            )

            context = zmq.Context()

            if self.molmo_reaching:
                self.molmo_socket = context.socket(zmq.REQ)
                self.molmo_socket.connect("tcp://localhost:5559")
                self.text_prompt = prompt = (
                    "Mark a point on the white USB port. Look at the lower region. "
                    'Output only one (x,y) relative coordinate in JSON format, e.g. {"x": 0.123, "y": 0.456}.'
                )

            if self.mask_view:
                # ZMQ client sockets
                self.dift_socket = context.socket(zmq.REQ)
                self.dift_socket.connect("tcp://localhost:5556")
                self.sam2_socket = context.socket(zmq.REQ)
                self.sam2_socket.connect("tcp://localhost:5557")
                self.xmem_socket = context.socket(zmq.REQ)
                self.xmem_socket.connect("tcp://localhost:5558")

                base_dir = Path("/mnt/robotlab/zifan/visk_rl_jax/plug_insertion_base")
                self.original_feature = torch.load(
                    base_dir / "insertion_original.pt"
                ).squeeze(0)
                self.socket_mask = cv2.imread(
                    str(base_dir / "original_mask_socket.png"), 0
                )
                self.socket_mask = cv2.resize(
                    self.socket_mask, (768, 768), interpolation=cv2.INTER_LINEAR
                )
                self.plug_mask = cv2.imread(str(base_dir / "original_mask_plug.png"), 0)
                self.plug_mask = cv2.resize(
                    self.plug_mask, (768, 768), interpolation=cv2.INTER_LINEAR
                )
                self.gripper_mask = cv2.imread(str(base_dir / "gripper_mask.png"), 0)

                self.gripper_mask = cv2.resize(
                    self.gripper_mask, (768, 768), interpolation=cv2.INTER_LINEAR
                )
                self.prev_mask = None

        else:
            self.pixel_keys_idx = []
            for cam_idx in list(CAM_SERIAL_NUMS.keys()) + list(
                FISH_EYE_CAM_SERIAL_NUMS.keys()
            ):
                self.pixel_keys_idx.append(cam_idx)

    def step(self, action):
        if self.use_robot:
            action = np.array(action)

            action_dict = {
                "xarm": {
                    "cartesian": action[:-1],
                    "gripper": action[-1:],
                }
            }

            # send action
            self.action_request_socket.send(pickle.dumps(action_dict, protocol=-1))
            ret = self.action_request_socket.recv()
            ret = pickle.loads(ret)
            if ret == "Command failed!":
                print("Command failed!")
                # return None, 0, True, None
                self.action_request_socket.send(b"get_state")
                ret = pickle.loads(self.action_request_socket.recv())
            #     robot_state = pickle.loads(self.action_request_socket.recv())["robot_state"]["xarm"]
            # else:
            #     # robot_state = ret["robot_state"]["xarm"]
            #     robot_state = ret["robot_state"]["xarm"]
            robot_state = ret["robot_state"]["xarm"]

            # cartesian_pos = robot_state[:3]
            # cartesian_ori = robot_state[3:6]
            # gripper = robot_state[6]
            # cartesian_ori_sin = np.sin(cartesian_ori)
            # cartesian_ori_cos = np.cos(cartesian_ori)
            # robot_state = np.concatenate(
            #     [cartesian_pos, cartesian_ori_sin, cartesian_ori_cos, [gripper]], axis=0
            # )
            cartesian = robot_state[:6]
            quat_cartesian = get_quaternion_orientation(cartesian)
            robot_state = np.concatenate([quat_cartesian, robot_state[6:]], axis=0)

            # subscribe images
            image_dict = {}
            depth_dict = {}
            for cam_idx, img_sub in self.image_subscribers.items():
                image_dict[cam_idx] = img_sub.recv_rgb_image()[0]
            # for cam_idx, img_sub in self.depth_subscribers.items():
            #     depth_dict[cam_idx] = img_sub.recv_depth_image()[0]

            obs = {}
            obs["features"] = np.array(robot_state, dtype=np.float32)
            obs["proprioceptive"] = np.array(robot_state, dtype=np.float32)
            if self.sensor_type == "reskin":
                try:
                    sensor_state = ret["sensor_state"]["reskin"]["sensor_values"]
                    sensor_history = np.array(
                        ret["sensor_state"]["reskin"]["sensor_history"]
                    )
                    sensor_state_sub = (
                        np.array(sensor_state, dtype=np.float32) - self.sensor_baseline
                    )
                    sensor_history_sub = (
                        np.array(sensor_history, dtype=np.float32)
                        - self.sensor_baseline[None, :]
                    )
                    sensor_diff = sensor_state_sub - self.sensor_prev_state
                    self.sensor_prev_state = sensor_state_sub
                    if self.separate_sensors:
                        sensor_keys = [
                            f"sensor{sensor_idx}"
                            for sensor_idx in range(self.n_sensors)
                        ]
                    else:
                        sensor_keys = ["sensor"]
                    for sidx, sensor_key in enumerate(sensor_keys):
                        if self.subtract_sensor_baseline:
                            obs[sensor_key] = sensor_state_sub[
                                sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
                            ]
                            obs[f"{sensor_key}_history"] = sensor_history_sub[
                                :, sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
                            ]
                        else:
                            obs[sensor_key] = sensor_state[
                                sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
                            ]
                            obs[f"{sensor_key}_history"] = sensor_history[
                                :, sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
                            ]
                        if self.use_sensor_diffs:
                            obs[sensor_key] = np.concatenate(
                                (
                                    obs[sensor_key],
                                    sensor_diff[
                                        sidx * self.sensor_dim : (sidx + 1)
                                        * self.sensor_dim
                                    ],
                                ),
                                axis=0,
                            )
                except KeyError:
                    pass
            elif self.sensor_type == "digit":
                for dkey in self.digit_keys:
                    obs[dkey] = np.array(ret["sensor_state"][dkey])
                    obs[dkey] = cv2.resize(obs[dkey], (self.width, self.height))
                    if self.subtract_sensor_baseline:
                        obs[dkey] = obs[dkey] - self.sensor_baseline

            for cam_idx, image in image_dict.items():
                if cam_idx == 52:
                    # crop the right side of the image for the gripper cam
                    img_shape = image.shape
                    crop_percent = 0.2
                    image = image[:, : int(img_shape[1] * (1 - crop_percent))]
                    obs[f"pixels{cam_idx}"] = cv2.resize(
                        image, (self.width, self.height)
                    )
                else:
                    obs[f"pixels{cam_idx}"] = image

            if self.mask_view:
                frame = cv2.resize(
                    obs["pixels52"], (768, 768), interpolation=cv2.INTER_LINEAR
                )

                current_mask = self.track_mask(frame, self.prev_mask, first_frame=False)
                self.prev_mask = current_mask

                combined_mask = np.maximum(
                    np.maximum(current_mask, self.gripper_mask), self.plug_mask
                )
                frame[combined_mask == 0] = 0
                obs["pixels52"] = cv2.resize(frame, (self.width, self.height))

                sanity_dir = Path("/mnt/robotlab/zifan/visk_rl_jax/sanity_check")
                sanity_dir.mkdir(exist_ok=True, parents=True)

                masked_img_path = sanity_dir / "masked_frame_step.png"

                cv2.imwrite(str(masked_img_path), frame)

            return obs, self.reward, False, self.done, {}

        else:
            # generate random observation and a dummy reward.
            print("current step's observation and reward is randomly generated!")
            obs = {}
            obs["features"] = np.random.rand(self.feature_dim).astype(np.float32)
            obs["proprioceptive"] = np.random.rand(self.proprio_dim).astype(np.float32)
            if not self.separate_sensors:
                obs["sensor"] = np.random.uniform(
                    0, 1, size=(self.sensor_dim * (1 + int(self.use_sensor_diffs)),)
                ).astype(np.float32)
            else:
                for sensor_idx in range(self.n_sensors):
                    obs[f"sensor{sensor_idx}"] = np.random.uniform(
                        0,
                        1,
                        size=(self.per_sensor_dim * (1 + int(self.use_sensor_diffs)),),
                    ).astype(np.float32)
            for cam_idx in self.pixel_keys_idx:
                obs[f"pixels{cam_idx}"] = np.random.randint(
                    0, 256, (self.height, self.width, self.n_channels), dtype=np.uint8
                )
            obs["goal_achieved"] = False
            reward = 0.5
            done = self.done
            info = {}
            return obs, reward, False, done, info

    def reset(self, seed=None):  # currently same positions, with gripper opening
        if self.use_robot:
            print("resetting")
            self.done = False
            self.action_request_socket.send(b"reset")
            reset_state = pickle.loads(self.action_request_socket.recv())

            # subscribe robot state
            self.action_request_socket.send(b"get_state")
            ret = pickle.loads(self.action_request_socket.recv())

            robot_state = ret["robot_state"]["xarm"]
            # robot_state = np.array(robot_state, dtype=np.float32)
            # cartesian_pos = robot_state[:3]
            # cartesian_ori = robot_state[3:6]
            # gripper = robot_state[6]
            # cartesian_ori_sin = np.sin(cartesian_ori)
            # cartesian_ori_cos = np.cos(cartesian_ori)
            # robot_state = np.concatenate(
            #     [cartesian_pos, cartesian_ori_sin, cartesian_ori_cos, [gripper]], axis=0
            # )
            cartesian = robot_state[:6]
            quat_cartesian = get_quaternion_orientation(cartesian)
            robot_state = np.concatenate([quat_cartesian, robot_state[6:]], axis=0)

            # subscribe images
            image_dict = {}
            depth_dict = {}

            for cam_idx, img_sub in self.image_subscribers.items():
                image_dict[cam_idx] = img_sub.recv_rgb_image()[0]
            for cam_idx, img_sub in self.depth_subscribers.items():
                depth_dict[cam_idx] = img_sub.recv_depth_image()[0]

            obs = {}
            obs["features"] = robot_state
            obs["proprioceptive"] = robot_state
            if self.sensor_type == "reskin":
                try:
                    sensor_state = np.array(
                        ret["sensor_state"]["reskin"]["sensor_values"]
                    )
                    # obs["sensor"] = np.array(sensor_state)
                    sensor_history = np.array(
                        ret["sensor_state"]["reskin"]["sensor_history"]
                    )
                    if self.subtract_sensor_baseline:
                        baseline_meas = []
                        while len(baseline_meas) < 5:
                            self.action_request_socket.send(b"get_sensor_state")
                            ret = pickle.loads(self.action_request_socket.recv())
                            sensor_state = ret["reskin"]["sensor_values"]
                            baseline_meas.append(sensor_state)
                        self.sensor_baseline = np.median(baseline_meas, axis=0)
                        sensor_state = sensor_state - self.sensor_baseline
                        sensor_history = sensor_history - self.sensor_baseline
                        # obs["sensor"] = sensor_state - self.sensor_baseline
                    # self.sensor_prev_state = obs["sensor"]
                    self.sensor_prev_state = sensor_state
                    if self.separate_sensors:
                        sensor_keys = [
                            f"sensor{sensor_idx}"
                            for sensor_idx in range(self.n_sensors)
                        ]
                    else:
                        sensor_keys = ["sensor"]
                    for sidx, sensor_key in enumerate(sensor_keys):
                        obs[sensor_key] = sensor_state[
                            sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
                        ]
                        obs[f"{sensor_key}_history"] = sensor_history[
                            :, sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
                        ]
                        if self.use_sensor_diffs:
                            obs[sensor_key] = np.concatenate(
                                (obs[sensor_key], np.zeros_like(obs[sensor_key]))
                            )
                except KeyError:
                    pass
            elif self.sensor_type == "digit":
                for dkey in self.digit_keys:
                    obs[dkey] = np.array(ret["sensor_state"][dkey])
                    obs[dkey] = cv2.resize(obs[dkey], (self.width, self.height))
                    if self.subtract_sensor_baseline:
                        baseline_meas = []
                        while len(baseline_meas) < 5:
                            self.action_request_socket.send(b"get_sensor_state")
                            ret = pickle.loads(self.action_request_socket.recv())
                            sensor_state = cv2.resize(
                                ret[dkey], (self.width, self.height)
                            )
                            baseline_meas.append(sensor_state)
                        self.sensor_baseline = np.median(baseline_meas, axis=0)
                        obs[dkey] = sensor_state - self.sensor_baseline
                        # obs["sensor"] = sensor_state - self.sensor_baseline
            for cam_idx, image in image_dict.items():
                if cam_idx == 52:
                    # crop the right side of the image for the gripper cam
                    img_shape = image.shape
                    crop_percent = 0.2
                    image = image[:, : int(img_shape[1] * (1 - crop_percent))]
                    obs[f"pixels{cam_idx}"] = cv2.resize(
                        image, (self.width, self.height)
                    )
                else:
                    obs[f"pixels{cam_idx}"] = image

            for cam_idx, depth_map in depth_dict.items():
                obs[f"depth{cam_idx}"] = depth_map

            print("returing")

            if self.mask_view:
                frame = cv2.resize(
                    obs["pixels52"], (768, 768), interpolation=cv2.INTER_LINEAR
                )

                features = self.get_features(frame)
                socket_center = (
                    np.array(np.where(self.socket_mask > 0)).mean(axis=1).astype(int)
                )
                scale_factor = 768 // 48
                fy, fx = socket_center // scale_factor

                orig_feat_vec = self.original_feature[:, fy, fx]
                orig_feat_vec = orig_feat_vec / np.linalg.norm(orig_feat_vec)

                curr_feat_flat = features.reshape(
                    1280, -1
                ).copy()  # fix read-only issue
                curr_feat_flat = curr_feat_flat / np.linalg.norm(
                    curr_feat_flat, axis=0, keepdims=True
                )

                cos_sim = (orig_feat_vec[:, None] * curr_feat_flat).sum(axis=0)
                max_idx = cos_sim.argmax()
                y, x = divmod(int(max_idx), 48)
                socket_point = [x * scale_factor, y * scale_factor]

                # Generate initial SAM2 mask for socket
                socket_mask = self.get_mask(frame, socket_point)
                combined_mask = np.maximum(
                    np.maximum(socket_mask, self.gripper_mask), self.plug_mask
                )
                self.prev_mask = socket_mask
                self.track_mask(frame, self.prev_mask, first_frame=True)

                # Apply mask to frame
                masked_frame = frame.copy()
                masked_frame[combined_mask == 0] = 0
                obs["pixels52"] = cv2.resize(masked_frame, (self.width, self.height))

                # Save the masked image and point for sanity check
                sanity_dir = Path("/mnt/robotlab/zifan/visk_rl_jax/sanity_check")
                sanity_dir.mkdir(exist_ok=True, parents=True)

                masked_img_path = sanity_dir / "masked_frame_reset.png"
                socket_point_path = sanity_dir / "socket_point_reset.txt"

                cv2.circle(
                    masked_frame, tuple(socket_point), 8, (0, 255, 0), thickness=-1
                )
                cv2.imwrite(str(masked_img_path), masked_frame)

                with open(socket_point_path, "w") as f:
                    f.write(f"{socket_point[0]}, {socket_point[1]}")

                print(f"Saved masked image for sanity check: {masked_img_path}")
                print(f"Socket point for sanity check: {socket_point}")

            return obs
        else:
            obs = {}
            obs["features"] = np.random.rand(self.feature_dim).astype(np.float32)
            obs["proprioceptive"] = np.random.rand(self.proprio_dim).astype(np.float32)
            if not self.separate_sensors:
                obs["sensor"] = np.random.uniform(
                    0, 1, size=(self.sensor_dim * (1 + int(self.use_sensor_diffs)),)
                ).astype(np.float32)
                # TODO: Add dummy variable for sensor history
                self.sensor_prev_state = obs["sensor"]
            else:
                for sensor_idx in range(self.n_sensors):
                    obs[f"sensor{sensor_idx}"] = np.random.uniform(
                        0,
                        1,
                        size=(self.per_sensor_dim * (1 + int(self.use_sensor_diffs)),),
                    ).astype(np.float32)
            self.sensor_baseline = np.zeros(self.per_sensor_dim * self.n_sensors)
            for cam_idx in self.pixel_keys_idx:
                obs[f"pixels{cam_idx}"] = np.random.randint(
                    0, 256, (self.height, self.width, self.n_channels), dtype=np.uint8
                )
            return obs

    def render(self, mode="rgb_array", width=640, height=480):
        print("rendering")
        # subscribe images
        image_list = []
        for _, img_sub in self.image_subscribers.items():
            image = img_sub.recv_rgb_image()[0]
            image_list.append(cv2.resize(image, (width, height)))

        obs = np.concatenate(image_list, axis=1)
        return obs

    def get_features(self, frame):
        _, frame_encoded = cv2.imencode(".jpg", frame)
        self.dift_socket.send(frame_encoded.tobytes())
        features = self.dift_socket.recv()
        return np.frombuffer(features, dtype=np.float32).reshape((1280, 48, 48))

    def get_mask(self, frame, point):
        _, frame_encoded = cv2.imencode(".jpg", frame)
        self.sam2_socket.send_multipart(
            [frame_encoded.tobytes(), np.array(point, dtype=np.int32).tobytes()]
        )
        mask = self.sam2_socket.recv()
        return np.frombuffer(mask, dtype=np.uint8).reshape((768, 768))

    def track_mask(self, frame, prev_mask, first_frame=False):
        _, frame_encoded = cv2.imencode(".jpg", frame)
        self.xmem_socket.send_multipart(
            [
                frame_encoded.tobytes(),
                prev_mask.tobytes(),
                str(int(first_frame)).encode(),
            ]
        )
        mask = self.xmem_socket.recv()
        return np.frombuffer(mask, dtype=np.uint8).reshape(frame.shape[:2])

    def send_reaching_prompt(self, text_prompt, image_data):
        # Send multipart message [text, image]
        self.molmo_socket.send_multipart([text_prompt.encode(), image_data])
        return self.molmo_socket.recv_string()


if __name__ == "__main__":
    env = RobotEnv()
    obs = env.reset()
    import ipdb

    ipdb.set_trace()

    for i in range(30):
        action = obs["features"]
        action[0] += 2
        obs, reward, done, _ = env.step(action)

    for i in range(30):
        action = obs["features"]
        action[1] += 2
        obs, reward, done, _ = env.step(action)

    for i in range(30):
        action = obs["features"]
        action[2] += 2
        obs, reward, done, _ = env.step(action)
