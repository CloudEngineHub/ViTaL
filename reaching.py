from pathlib import Path
from io import BytesIO
import zmq
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class MolmoZMQClient:
    def __init__(self, server_address="localhost:6666"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_address}")

    def send_cropped_prompt(self, text_prompt: str, img: np.ndarray) -> (str, tuple):
        # img is H×W×3

        h, w = img.shape[:2]
        left, right = int(0.4 * w), int(0.8 * w)
        top, bottom = int(0.7 * h), h

        cw, ch = right - left, bottom - top
        cropped_np = img[top:bottom, left:right, :]
        cropped = Image.fromarray(cropped_np)

        buf = BytesIO()
        cropped.save(buf, format="JPEG")

        self.socket.send_multipart([text_prompt.encode(), buf.getvalue()])
        return self.socket.recv_string(), (left, top, cw, ch)
