from functools import partial

import torch
import numpy as np


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def get_unique_labels(mask):
    return np.nonzero(np.bincount(mask.flatten() + 1))[0] - 1


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (
        max(rmin, bbox[0]),
        min(rmax, bbox[1]),
        max(cmin, bbox[2]),
        min(cmax, bbox[3]),
    )


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union
