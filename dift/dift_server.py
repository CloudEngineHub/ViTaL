import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import zmq
import torch
from PIL import Image
import io
from src.models.dift_sd import SDFeaturizer
from torchvision.transforms import PILToTensor


class DIFTServer:
    def __init__(self, port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

    def extract_features(self, image_bytes):
        # Initialize the model on every call to save memory usage for Online RL.
        dift = SDFeaturizer("stabilityai/stable-diffusion-2-1")

        image = Image.open(io.BytesIO(image_bytes)).resize((768, 768))
        img_tensor = (PILToTensor()(image).float().cuda() / 255.0 - 0.5) * 2

        features = dift.forward(
            img_tensor, prompt="", t=261, up_ft_index=1, ensemble_size=1
        )
        features = features.squeeze(0).cpu().numpy()

        # Clean up and free GPU memory.
        del dift, img_tensor
        torch.cuda.empty_cache()

        return features.tobytes()

    def run(self):
        print("DIFT Server started...")
        while True:
            image_bytes = self.socket.recv()
            features = self.extract_features(image_bytes)
            del image_bytes
            self.socket.send(features)


if __name__ == "__main__":
    server = DIFTServer(port=5556)
    server.run()
