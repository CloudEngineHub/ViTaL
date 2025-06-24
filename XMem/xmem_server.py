import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import zmq
import torch
import numpy as np
import cv2
from inference.inference_core import InferenceCore
from model.network import XMem


class XMemServer:
    def __init__(self, model_path, port=5558):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        self.config = {
            "model": model_path,
            "dataset": "G",
            "split": "val",
            "save_all": True,
            "benchmark": False,
            "disable_long_term": False,
            "max_mid_term_frames": 10,
            "min_mid_term_frames": 5,
            "max_long_term_elements": 10000,
            "num_prototypes": 128,
            "top_k": 30,
            "mem_every": 5,
            "deep_update_every": -1,
            "save_scores": False,
            "flip": False,
            "enable_long_term": True,
            "key_dim": 64,
            "value_dim": 512,
            "hidden_dim": 64,
            "enable_long_term_count_usage": False,
        }
        self.network = XMem({}, model_path).cuda().eval()
        self.network.load_weights(torch.load(model_path), init_as_zero_if_needed=True)
        self.processor = None  # Initialize on first frame.

    def generate_mask(self, frame_bytes, prev_mask_bytes, first_frame=False):
        frame_np = cv2.imdecode(
            np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        frame_np = cv2.resize(frame_np, (768, 768), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor and normalize.
        frame_tensor = (
            torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float().cuda()
            / 255.0
        )

        with torch.no_grad():
            if first_frame:
                torch.cuda.empty_cache()
                mask_np = np.frombuffer(prev_mask_bytes, dtype=np.uint8).reshape(
                    frame_np.shape[:2]
                )
                mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0).cuda()
                self.processor = InferenceCore(self.network, self.config)
                self.processor.set_all_labels([1])
                prob = self.processor.step(frame_tensor[0], mask_tensor, [1])
                del mask_tensor, mask_np

            else:
                prob = self.processor.step(frame_tensor[0])

        # Convert probability map to a binary mask.
        mask_pred = torch.argmax(prob, dim=0).cpu().numpy().astype(np.uint8) * 255

        del frame_np, frame_tensor, prob
        torch.cuda.empty_cache()

        return mask_pred.tobytes()

    def run(self):
        print("XMem Server started...")
        while True:
            msg_parts = self.socket.recv_multipart()
            frame_bytes = msg_parts[0]
            prev_mask_bytes = msg_parts[1]
            first_frame = bool(int(msg_parts[2].decode()))
            mask_bytes = self.generate_mask(frame_bytes, prev_mask_bytes, first_frame)
            self.socket.send(mask_bytes)
            del mask_bytes, first_frame, prev_mask_bytes, frame_bytes
            torch.cuda.empty_cache()


if __name__ == "__main__":
    server = XMemServer(model_path="./saves/XMem.pth", port=5558)
    server.run()
