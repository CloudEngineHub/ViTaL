import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import zmq
import torch
from PIL import Image
import numpy as np
import io
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Server:
    def __init__(self, port=5557):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

    def generate_mask(self, image_bytes, coords_bytes, labels_bytes):
        image = Image.open(io.BytesIO(image_bytes)).resize((768, 768))
        image_np = np.array(image)
        predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

        point_coords = np.frombuffer(coords_bytes, dtype=np.int32).reshape(-1, 2)
        point_labels = np.frombuffer(labels_bytes, dtype=np.int32)  # shape (N,)

        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            point_coords=point_coords, point_labels=point_labels, multimask_output=False
        )

        mask = (masks[0] * 255).astype(np.uint8)
        del predictor
        torch.cuda.empty_cache()
        return mask.tobytes()

    def run(self):
        print("SAM2 Server started...")
        while True:
            msg_parts = self.socket.recv_multipart()
            # image_bytes = msg_parts[0]
            # point_coords = np.frombuffer(msg_parts[1], dtype=np.int32).tolist()
            # labels = np.frombuffer(msg_parts[2], dtype=np.int32).tolist()
            mask_bytes = self.generate_mask(msg_parts[0], msg_parts[1], msg_parts[2])
            del msg_parts  # image_bytes, point_coords,labels
            torch.cuda.empty_cache()
            self.socket.send(mask_bytes)


if __name__ == "__main__":
    server = SAM2Server(port=5557)
    server.run()
