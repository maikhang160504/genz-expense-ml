"""Geometry + filters for page skew / 0-180 rotation (from MC_OCR rotation_corrector)."""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.spatial import ConvexHull


def _euclidean(pt1: list[float], pt2: list[float]) -> float:
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


class TextPoly:
    def __init__(self, segment_pts: list[int] | str):
        if isinstance(segment_pts, str):
            segment_pts = [int(f) for f in segment_pts.split(",")]
        else:
            segment_pts = [round(float(f)) for f in segment_pts]
        num_pts = len(segment_pts) // 2
        self.list_pts = [[segment_pts[2 * i], segment_pts[2 * i + 1]] for i in range(num_pts)]

    def get_horizontal_angle(self) -> float:
        assert len(self.list_pts) == 4
        first_edge = _euclidean(self.list_pts[0], self.list_pts[1])
        second_edge = _euclidean(self.list_pts[1], self.list_pts[2])
        if first_edge / max(second_edge, 1e-6) > 1:
            long_edge = (
                self.list_pts[0][0] - self.list_pts[1][0],
                self.list_pts[0][1] - self.list_pts[1][1],
            )
        else:
            long_edge = (
                self.list_pts[1][0] - self.list_pts[2][0],
                self.list_pts[1][1] - self.list_pts[2][1],
            )
        if long_edge[0] == 0:
            return -90.0 if long_edge[1] < 0 else 90.0
        return math.atan2(long_edge[1], long_edge[0]) * 57.296


def drop_box(boxlist: list[Any], drop_gap: tuple[float, float] = (0.5, 2.0)) -> list[Any]:
    kept: list[Any] = []
    for box_data in boxlist:
        box = box_data["coors"] if isinstance(box_data, dict) else box_data
        box_np = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        _w, _h = cv2.minAreaRect(box_np)[1]
        if min(drop_gap) < _w / max(_h, 1e-6) < max(drop_gap):
            continue
        kept.append(box_data)
    return kept


def filter_outliers_angle(list_angle: list[float], thresh: float = 45.0) -> list[float]:
    arr = np.abs(np.array(list_angle, dtype=np.float64))
    if arr.max() - arr.min() <= thresh:
        return list_angle
    codebook, _ = kmeans(arr, 2)
    cluster_indices, _ = vq(arr, codebook)
    buckets: dict[int, list[float]] = {}
    for idx, val in enumerate(arr):
        buckets.setdefault(int(cluster_indices[idx]), []).append(float(val))
    largest = max(buckets.values(), key=len)
    return largest


def get_mean_horizontal_angle(boxlist: list[Any], cluster: bool = True) -> float:
    if not boxlist:
        return 0.0
    angles: list[float] = []
    for box_data in boxlist:
        box = box_data["coors"] if isinstance(box_data, dict) else box_data
        angle = TextPoly(box).get_horizontal_angle()
        if angle >= 0:
            angle = 180 - angle + 90
        else:
            angle = abs(angle) - 90
        angles.append(angle)
    if cluster:
        angles = filter_outliers_angle(angles)
    mean_angle = float(np.mean(angles)) - 90.0
    return mean_angle


def filter_90_box(boxlist: list[Any], thresh: float = 45.0) -> list[Any]:
    if not boxlist:
        return boxlist
    angles: list[float] = []
    for box_data in boxlist:
        box = box_data["coors"] if isinstance(box_data, dict) else box_data
        angles.append(abs(TextPoly(box).get_horizontal_angle()))
    arr = np.array(angles)
    if arr.max() - arr.min() <= thresh:
        return boxlist
    codebook, _ = kmeans(arr, 2)
    cluster_indices, _ = vq(arr, codebook)
    buckets: dict[int, list[Any]] = {}
    for idx, box_data in enumerate(boxlist):
        buckets.setdefault(int(cluster_indices[idx]), []).append(box_data)
    return max(buckets.values(), key=len)


def rotate_image_bbox_angle(img: np.ndarray, bboxes: list[Any], angle: float) -> tuple[np.ndarray, list[Any]]:
    h_org, w_org = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w_org / 2, h_org / 2), 360 - angle, 1)
    rad = math.radians(angle)
    sin, cos = math.sin(rad), math.cos(rad)
    bound_w = int((h_org * abs(sin)) + (w_org * abs(cos)))
    bound_h = int((h_org * abs(cos)) + (w_org * abs(sin)))
    mat[0, 2] += max(0.0, ((bound_w / 2) - w_org / 2) - 1)
    mat[1, 2] += max(0.0, ((bound_h / 2) - h_org / 2) - 1)
    img_result = cv2.warpAffine(img, mat, (bound_w, bound_h))

    def _rotate_flat(flat: list[int]) -> list[int]:
        pts = np.array(flat, dtype=np.float64).reshape(-1, 2)
        ones = np.ones((len(pts), 1))
        transformed = mat.dot(np.hstack([pts, ones]).T).T
        return [int(round(v)) for v in transformed.reshape(-1)]

    ret_boxes: list[Any] = []
    for box_data in bboxes:
        if isinstance(box_data, dict):
            ret_boxes.append({"coors": _rotate_flat(box_data["coors"]), "data": box_data.get("data", "")})
        else:
            ret_boxes.append(_rotate_flat(box_data))
    return img_result, ret_boxes


def rotate_and_crop(
    img: np.ndarray,
    points: np.ndarray,
    *,
    rotate: bool = True,
    extend: bool = True,
    extend_x_ratio: float = 0.0001,
    extend_y_ratio: float = 0.0001,
    min_extend_y: int = 2,
    min_extend_x: int = 1,
) -> np.ndarray:
    rect = cv2.minAreaRect(points)
    box = np.int32(cv2.boxPoints(rect))
    height = int(rect[1][0])
    width = int(rect[1][1])
    if extend:
        w, h = (width, height) if width > height else (height, width)
        ex = min_extend_x if (extend_x_ratio * w) < min_extend_x else int(round(extend_x_ratio * w))
        ey = min_extend_y if (extend_y_ratio * h) < min_extend_y else int(round(extend_y_ratio * h))
        if width < height:
            ex, ey = ey, ex
    else:
        ex = ey = 0
    src_pts = box.astype("float32")
    dst_pts = np.array(
        [
            [width - 1 + ex, height - 1 + ey],
            [ex, height - 1 + ey],
            [ex, ey],
            [width - 1 + ex, ey],
        ],
        dtype="float32",
    )
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src_pts, dst_pts), (width + 2 * ex, height + 2 * ey))
    h, w, _ = warped.shape
    if w < h and rotate:
        return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def vote_page_flip(model: Any, img_bgr: np.ndarray, boxes_list: list[Any]) -> int:
    """Return 0 or 180 degrees to rotate the full page."""
    counts = {"0": 0, "180": 0}
    for box_data in boxes_list:
        box = box_data["coors"] if isinstance(box_data, dict) else box_data
        box_loc = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        crop = rotate_and_crop(
            img_bgr,
            box_loc,
            extend=True,
            extend_x_ratio=0.0001,
            extend_y_ratio=0.0001,
            min_extend_y=2,
            min_extend_x=1,
        )
        label, _conf = model.classify_crop(crop)
        counts[label] = counts.get(label, 0) + 1
    return 0 if counts["0"] >= counts["180"] else 180
