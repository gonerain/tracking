from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]

INSTANCE_PALETTE = [
    np.array([255, 80, 80], dtype=np.uint8),
    np.array([80, 255, 80], dtype=np.uint8),
    np.array([80, 80, 255], dtype=np.uint8),
    np.array([255, 200, 80], dtype=np.uint8),
    np.array([255, 80, 200], dtype=np.uint8),
    np.array([80, 255, 255], dtype=np.uint8),
    np.array([200, 120, 255], dtype=np.uint8),
    np.array([255, 160, 120], dtype=np.uint8),
]


def build_detectron_predictor(conf_file: str, model_file: str, task: str = "semantic") -> Any:
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    from imseg.mask2former.mask2former import add_maskformer2_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(conf_file)
    task = str(task).lower()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = task == "semantic"
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = task == "instance"
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = task == "panoptic"
    cfg.merge_from_list(["MODEL.WEIGHTS", model_file])
    cfg.freeze()
    return DefaultPredictor(cfg)


def get_predict_func_detectron(conf_file: str, model_file: str):
    predictor_base = build_detectron_predictor(conf_file, model_file, task="semantic")

    def predictor(img: np.ndarray) -> np.ndarray:
        simg = predictor_base(img)["sem_seg"]
        simg[simg < 0.5] = 0
        return simg.argmax(axis=0).cpu().numpy().astype("uint8")

    return predictor


def get_predict_func_detectron_instance(conf_file: str, model_file: str):
    predictor_base = build_detectron_predictor(conf_file, model_file, task="instance")

    def predictor(img: np.ndarray) -> dict[str, np.ndarray]:
        output = predictor_base(img)
        instances = output["instances"].to("cpu")
        pred_masks = instances.pred_masks.numpy() if instances.has("pred_masks") else np.zeros((0, img.shape[0], img.shape[1]), dtype=bool)
        pred_classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros((0,), dtype=np.int64)
        scores = instances.scores.numpy() if instances.has("scores") else np.zeros((0,), dtype=np.float32)
        return {
            "pred_masks": pred_masks,
            "pred_classes": pred_classes,
            "scores": scores,
        }

    return predictor


def get_predict_func_detectron_panoptic(conf_file: str, model_file: str):
    predictor_base = build_detectron_predictor(conf_file, model_file, task="panoptic")

    def predictor(img: np.ndarray) -> dict[str, Any]:
        output = predictor_base(img)
        panoptic_seg, segments_info = output["panoptic_seg"]
        return {
            "panoptic_seg": panoptic_seg.to("cpu").numpy().astype(np.int32),
            "segments_info": segments_info,
        }

    return predictor


def get_predict_func_mmsegmentation(conf_file: str, model_file: str):
    from mmseg.apis import inference_segmentor, init_segmentor

    model = init_segmentor(conf_file, model_file, device="cuda:0")

    def predictor(img: np.ndarray) -> np.ndarray:
        return inference_segmentor(model, img)[0]

    return predictor


class SegmentationPredictor:
    def __init__(self, config: dict[str, Any]) -> None:
        seg_cfg = config.get("segmentation", {})
        self.enabled = bool(seg_cfg.get("enabled", True))
        self.person_class_id = int(seg_cfg.get("person_class_id", 19))
        self.predictor_kind = str(seg_cfg.get("predictor_kind", "detectron")).lower()
        self.task = str(seg_cfg.get("task", "semantic")).lower()
        self.config_path = Path(str(seg_cfg.get("config_path", ROOT / "imseg" / "mask2former" / "config" / "swin" / "maskformer2_swin_large_IN21k_384_bs16_300k.yaml")))
        self.model_path = Path(str(seg_cfg.get("model_path", ROOT / "imseg" / "mask2former" / "model" / "model.pkl")))
        self._predictor = None
        self._load_error: str | None = None
        if not self.enabled:
            return

        try:
            if self.predictor_kind == "mmseg":
                factory = get_predict_func_mmsegmentation
            elif self.task == "instance":
                factory = get_predict_func_detectron_instance
            elif self.task == "panoptic":
                factory = get_predict_func_detectron_panoptic
            else:
                factory = get_predict_func_detectron
            self._predictor = factory(str(self.config_path), str(self.model_path))
        except Exception as exc:
            self._load_error = repr(exc)

    @property
    def available(self) -> bool:
        return self.enabled and self._predictor is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def predict(self, frame: np.ndarray) -> Any:
        if self._predictor is None:
            return None
        return self._predictor(frame)


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def visualize_semantic(frame: np.ndarray, segmentation: np.ndarray, person_class_id: int) -> np.ndarray:
    vis = frame.copy()
    person_mask = segmentation == int(person_class_id)
    overlay = vis.copy()
    overlay[person_mask] = np.array([0, 0, 255], dtype=np.uint8)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    return vis


def visualize_instance(frame: np.ndarray, prediction: dict[str, np.ndarray], person_class_id: int) -> np.ndarray:
    vis = frame.copy()
    masks = extract_person_masks(prediction, person_class_id, split_mode="none")
    for index, mask in enumerate(masks):
        if not np.any(mask):
            continue
        color = INSTANCE_PALETTE[index % len(INSTANCE_PALETTE)]
        overlay = vis.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color.tolist(), 2)
    return vis


def visualize_panoptic(frame: np.ndarray, prediction: dict[str, Any], person_class_id: int) -> np.ndarray:
    vis = frame.copy()
    masks = extract_person_masks(prediction, person_class_id, split_mode="none")
    for index, mask in enumerate(masks):
        color = INSTANCE_PALETTE[index % len(INSTANCE_PALETTE)]
        overlay = vis.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color.tolist(), 2)
    return vis


def extract_person_mask(prediction: Any, person_class_id: int, split_mode: str = "auto") -> np.ndarray | None:
    person_masks = extract_person_masks(prediction, person_class_id, split_mode=split_mode)
    if not person_masks:
        return None
    return np.any(np.stack(person_masks, axis=0), axis=0)


def extract_person_masks(prediction: Any, person_class_id: int, split_mode: str = "auto") -> list[np.ndarray]:
    if prediction is None:
        return []

    if isinstance(prediction, dict):
        if "pred_masks" in prediction:
            masks = prediction.get("pred_masks", np.zeros((0, 0, 0), dtype=bool))
            classes = prediction.get("pred_classes", np.zeros((0,), dtype=np.int64))
            person_masks: list[np.ndarray] = []
            for mask, cls in zip(masks, classes):
                if int(cls) != int(person_class_id):
                    continue
                mask_bool = np.asarray(mask, dtype=bool)
                if np.any(mask_bool):
                    person_masks.append(mask_bool)
            return person_masks

        if "panoptic_seg" in prediction:
            panoptic_seg = prediction.get("panoptic_seg")
            segments_info = prediction.get("segments_info", [])
            if panoptic_seg is None:
                return []
            person_masks: list[np.ndarray] = []
            for segment in segments_info:
                if int(segment.get("category_id", -1)) == int(person_class_id):
                    mask = panoptic_seg == int(segment["id"])
                    if np.any(mask):
                        person_masks.append(mask)
            return person_masks

        return []

    person_mask = np.asarray(prediction, dtype=np.int32) == int(person_class_id)
    if not np.any(person_mask):
        return []
    if str(split_mode).lower() == "none":
        return [person_mask]
    return _split_touching_instances(person_mask)


def select_relevant_person_masks(
    masks: list[np.ndarray],
    image_shape: tuple[int, ...],
    min_area_px: int = 0,
    max_instances: int | None = None,
) -> list[np.ndarray]:
    if not masks:
        return []

    height, width = int(image_shape[0]), int(image_shape[1])
    ranked: list[tuple[float, np.ndarray]] = []
    for mask in masks:
        mask = np.asarray(mask, dtype=bool)
        area = int(np.count_nonzero(mask))
        if area < int(min_area_px):
            continue
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        bottom = float(ys.max()) / max(height - 1, 1)
        center_x = float(xs.mean()) / max(width - 1, 1)
        center_bias = 1.0 - min(abs(center_x - 0.5) / 0.5, 1.0)
        area_ratio = float(area) / max(height * width, 1)
        score = 3.0 * area_ratio + 1.5 * bottom + 0.5 * center_bias
        ranked.append((score, mask))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [mask for _, mask in ranked]
    if max_instances is not None:
        selected = selected[: max(int(max_instances), 0)]
    return selected


def _split_touching_instances(mask: np.ndarray, min_area_px: int = 150, peak_rel_threshold: float = 0.45) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []

    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    instances: list[np.ndarray] = []

    for label_id in range(1, num_labels):
        component = labels == label_id
        area = int(np.count_nonzero(component))
        if area < min_area_px:
            continue

        dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist <= 1e-6:
            instances.append(component)
            continue

        peak_mask = (dist >= peak_rel_threshold * max_dist) & component
        peak_count, peak_labels = cv2.connectedComponents(peak_mask.astype(np.uint8))
        if peak_count <= 2:
            instances.append(component)
            continue

        centroids: list[np.ndarray] = []
        for peak_id in range(1, peak_count):
            ys, xs = np.where(peak_labels == peak_id)
            if xs.size == 0:
                continue
            centroids.append(np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32))

        if len(centroids) <= 1:
            instances.append(component)
            continue

        ys, xs = np.where(component)
        points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        centroid_array = np.stack(centroids, axis=0)
        distances = ((points[:, None, :] - centroid_array[None, :, :]) ** 2).sum(axis=2)
        assignments = np.argmin(distances, axis=1)

        for centroid_index in range(len(centroids)):
            submask = np.zeros_like(component, dtype=bool)
            selected = assignments == centroid_index
            if not np.any(selected):
                continue
            submask[ys[selected], xs[selected]] = True
            if np.count_nonzero(submask) >= min_area_px:
                instances.append(submask)

    return instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image segmentation using the project's configured model.")
    parser.add_argument("--config", default="configs/lidar_config.yaml", help="Path to YAML config containing segmentation settings.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", default="outputs/segmentation_preview.png", help="Output preview image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    predictor = SegmentationPredictor(config)
    if not predictor.available:
        raise RuntimeError(f"Segmentation predictor unavailable: {predictor.load_error}")

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    prediction = predictor.predict(image)
    if predictor.task == "instance":
        vis = visualize_instance(image, prediction, predictor.person_class_id)
    elif predictor.task == "panoptic":
        vis = visualize_panoptic(image, prediction, predictor.person_class_id)
    else:
        vis = visualize_semantic(image, prediction, predictor.person_class_id)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
    print(f"Wrote preview to {output_path}")


if __name__ == "__main__":
    main()
