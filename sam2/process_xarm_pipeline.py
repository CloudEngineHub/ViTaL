import os
import torch
import numpy as np
import cv2
from PIL import Image
import pickle
from sam2.sam2_image_predictor import SAM2ImagePredictor

TASK_NAME = ""
DATA_PATH = f"/path_to_data/{TASK_NAME}.pkl"
FEATURE_MAP_DIR = f"/path_to_data/bcrl_pipeline/{TASK_NAME}/dift/dift_feature_map/"
OUTPUT_DIR = f"/path_to_data/bcrl_pipeline/{TASK_NAME}/sam2/masked_image"
PROCESSED_MASK_DIR = f"/path_to_data/bcrl_pipeline/{TASK_NAME}/sam2/processed_mask"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_MASK_DIR, exist_ok=True)

TARGET_MASK_PATH = (
    f"/path_to_data/bcrl_pipeline/{TASK_NAME}/base/original_mask_target.png"
)
ORIGINAL_MASK_OBJECT_PATH = (
    f"/path_to_data/bcrl_pipeline/{TASK_NAME}/base/original_mask_object.png"
)
GRIPPER_MASK_PATH = f"/path_to_data/bcrl_pipeline/{TASK_NAME}/base/gripper_mask.png"

IMG_SIZE = (768, 768)
FEATURE_MAP_SIZE = (48, 48)
SCALE_FACTOR = IMG_SIZE[0] // FEATURE_MAP_SIZE[0]

raw_target = cv2.imread(TARGET_MASK_PATH, cv2.IMREAD_GRAYSCALE)
resized_mask = cv2.resize(raw_target, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
_, bin_mask = cv2.threshold(resized_mask, 1, 255, cv2.THRESH_BINARY)


def compute_quadrant_centers(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 2) split bbox into 4
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    mid_x = x_min + w // 2
    mid_y = y_min + h // 2

    quads = [
        (x_min, y_min, mid_x - 1, mid_y - 1),  # top‑left
        (mid_x, y_min, x_max, mid_y - 1),  # top‑right
        (x_min, mid_y, x_max // 2 + x_min - 1, y_max),  # bottom‑left
        (mid_x, mid_y, x_max, y_max),  # bottom‑right
    ]

    centers = []
    for x0, y0, x1, y1 in quads:
        sub = mask[y0 : y1 + 1, x0 : x1 + 1]
        ys_q, xs_q = np.where(sub > 0)
        if len(xs_q):
            # convert back into full‑image coords
            cx = int(xs_q.mean()) + x0
            cy = int(ys_q.mean()) + y0
            centers.append((cx, cy))
    return centers


def sample_mask_points(binary_mask, num_points=10):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty; no points to sample.")
    replace = len(xs) < num_points
    idx = np.random.choice(len(xs), size=num_points, replace=replace)
    return list(zip(xs[idx], ys[idx]))


def compute_mask_center(binary_mask):
    ys, xs = np.where(binary_mask > 0)
    return int(xs.mean()), int(ys.mean())


ft_original = torch.load(
    f"/path_to_data/bcrl_pipeline/{TASK_NAME}/base/dift_feature_map.pt"
).squeeze(0)
ft_orig_flat = ft_original.view(ft_original.shape[0], -1)
ft_orig_norm = ft_orig_flat / (ft_orig_flat.norm(dim=0, keepdim=True) + 1e-8)

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

for index, entry in enumerate(data["observations"]):
    first_frame = entry["pixels52"][0]
    original_size = (first_frame.shape[1], first_frame.shape[0])
    img = Image.fromarray(first_frame).resize(IMG_SIZE)
    img_np = np.array(img)

    feat_path = os.path.join(FEATURE_MAP_DIR, f"{index:04d}.pt")
    if not os.path.exists(feat_path):
        continue
    ft_current = torch.load(feat_path).squeeze(0)
    ft_curr_flat = ft_current.view(ft_current.shape[0], -1)
    ft_curr_norm = ft_curr_flat / (ft_curr_flat.norm(dim=0, keepdim=True) + 1e-8)

    mask_center = compute_mask_center(bin_mask)

    # random_pts = sample_mask_points(bin_mask, num_points=3)
    # original_pts = [mask_center] + random_pts

    global_mean = compute_mask_center(bin_mask)
    quadrant_means = compute_quadrant_centers(bin_mask)

    original_pts = [global_mean]  # + quadrant_means

    def find_corresponding_point(pt, orig_norm, curr_norm):
        fx, fy = pt[0] // SCALE_FACTOR, pt[1] // SCALE_FACTOR
        fx = np.clip(fx, 0, FEATURE_MAP_SIZE[0] - 1)
        fy = np.clip(fy, 0, FEATURE_MAP_SIZE[1] - 1)
        orig_vec = orig_norm[:, fy * FEATURE_MAP_SIZE[0] + fx].unsqueeze(0)
        cos_sim = torch.matmul(orig_vec, curr_norm).reshape(
            FEATURE_MAP_SIZE[1], FEATURE_MAP_SIZE[0]
        )
        y_idx, x_idx = np.unravel_index(torch.argmax(cos_sim).item(), cos_sim.shape)
        return x_idx * SCALE_FACTOR, y_idx * SCALE_FACTOR

    mapped_pts_tmp = [
        find_corresponding_point(p, ft_orig_norm, ft_curr_norm) for p in original_pts
    ]
    mapped_pts = []
    for i, (x, y) in enumerate(mapped_pts_tmp):
        mapped_pts.append((x, y))
    points = np.array(mapped_pts)
    labels = np.ones(len(points), dtype=int)

    if len(points) == 0:
        continue

    vis = img_np.copy()
    for x, y in points:
        cv2.circle(vis, (x, y), radius=8, color=(0, 255, 0), thickness=-1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"target_points_{index}.png"), vis)

    predictor.set_image(img_np)
    masks, _, _ = predictor.predict(
        point_coords=points, point_labels=labels, multimask_output=False
    )
    final_mask = (masks[0] * 255).astype(np.uint8)

    mask_final = (masks[0] * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    # mask_final = cv2.dilate( mask_final, kernel, iterations=1)

    plug_mask = cv2.imread(ORIGINAL_MASK_OBJECT_PATH, cv2.IMREAD_GRAYSCALE)
    plug_mask = cv2.resize(plug_mask, IMG_SIZE)

    gripper_mask = cv2.imread(GRIPPER_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    gripper_mask = cv2.resize(gripper_mask, IMG_SIZE)

    _, mask1_bin = cv2.threshold(mask_final, 1, 255, cv2.THRESH_BINARY)
    _, plug_mask_bin = cv2.threshold(plug_mask, 1, 255, cv2.THRESH_BINARY)
    _, gripper_mask_bin = cv2.threshold(gripper_mask, 1, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_or(
        cv2.bitwise_or(mask1_bin, plug_mask_bin), gripper_mask_bin
    )

    colored_mask = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.7, colored_mask, 0.3, 0
    )

    mask_resized = cv2.resize(
        combined_mask, original_size, interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_{index}.png"), mask_final)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"overlay_{index}.png"), overlay)
    cv2.imwrite(
        os.path.join(PROCESSED_MASK_DIR, f"processed_mask_{index}_resized.png"),
        mask_resized,
    )
    cv2.imwrite(os.path.join(PROCESSED_MASK_DIR, f"{index}/0000.png"), mask1_bin)

    print(f"Processed and saved results for demo {index}")
