import os
import pickle
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from src.models.dift_sd import SDFeaturizer

TASK_NAME = ""
# Path to your data pickle file
data_path = f"/path_to_data/{TASK_NAME}.pkl"
# Directory where extracted feature maps will be saved
output_dir = f"/path_to_data/bcrl_pipeline/{TASK_NAME}/dift/feature_map/"
os.makedirs(output_dir, exist_ok=True)

model_id = "stabilityai/stable-diffusion-2-1"
img_size = (768, 768)
t = 261
up_ft_index = 1
ensemble_size = 1
prompt = ""

dift = SDFeaturizer(model_id)

with open(data_path, "rb") as f:
    data = pickle.load(f)

observations = data["observations"]

for index, entry in enumerate(observations):
    frame = entry["pixels52"][0]
    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(frame.astype("uint8"))
    if img_size[0] > 0:
        frame = frame.resize(img_size)

    img_tensor = (PILToTensor()(frame) / 255.0 - 0.5) * 2
    ft = dift.forward(
        img_tensor,
        prompt=prompt,
        t=t,
        up_ft_index=up_ft_index,
        ensemble_size=ensemble_size,
    )

    ft = ft.squeeze(0).cpu()
    output_path = os.path.join(output_dir, f"{index:04d}.pt")
    torch.save(ft, output_path)
    print(f"Saved feature map for observation {index} to {output_path}")
