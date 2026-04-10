from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


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
    masks = prediction.get("pred_masks", np.zeros((0, frame.shape[0], frame.shape[1]), dtype=bool))
    classes = prediction.get("pred_classes", np.zeros((0,), dtype=np.int64))
    palette = [
        np.array([255, 80, 80], dtype=np.uint8),
        np.array([80, 255, 80], dtype=np.uint8),
        np.array([80, 80, 255], dtype=np.uint8),
        np.array([255, 200, 80], dtype=np.uint8),
        np.array([255, 80, 200], dtype=np.uint8),
        np.array([80, 255, 255], dtype=np.uint8),
    ]
    for index, (mask, cls) in enumerate(zip(masks, classes)):
        if int(cls) != int(person_class_id):
            continue
        overlay = vis.copy()
        overlay[mask] = palette[index % len(palette)]
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    return vis


def visualize_panoptic(frame: np.ndarray, prediction: dict[str, Any], person_class_id: int) -> np.ndarray:
    vis = frame.copy()
    panoptic_seg = prediction.get("panoptic_seg")
    segments_info = prediction.get("segments_info", [])
    if panoptic_seg is None:
        return vis
    palette = [
        np.array([255, 80, 80], dtype=np.uint8),
        np.array([80, 255, 80], dtype=np.uint8),
        np.array([80, 80, 255], dtype=np.uint8),
        np.array([255, 200, 80], dtype=np.uint8),
        np.array([255, 80, 200], dtype=np.uint8),
        np.array([80, 255, 255], dtype=np.uint8),
    ]
    for index, segment in enumerate(segments_info):
        if int(segment.get("category_id", -1)) != int(person_class_id):
            continue
        mask = panoptic_seg == int(segment["id"])
        overlay = vis.copy()
        overlay[mask] = palette[index % len(palette)]
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    return vis


def extract_person_mask(prediction: Any, person_class_id: int) -> np.ndarray | None:
    if prediction is None:
        return None

    if isinstance(prediction, dict):
        if "pred_masks" in prediction:
            masks = prediction.get("pred_masks", np.zeros((0, 0, 0), dtype=bool))
            classes = prediction.get("pred_classes", np.zeros((0,), dtype=np.int64))
            person_masks = [np.asarray(mask, dtype=bool) for mask, cls in zip(masks, classes) if int(cls) == int(person_class_id)]
            if not person_masks:
                return None
            return np.any(np.stack(person_masks, axis=0), axis=0)

        if "panoptic_seg" in prediction:
            panoptic_seg = prediction.get("panoptic_seg")
            segments_info = prediction.get("segments_info", [])
            if panoptic_seg is None:
                return None
            person_mask = np.zeros_like(panoptic_seg, dtype=bool)
            for segment in segments_info:
                if int(segment.get("category_id", -1)) == int(person_class_id):
                    person_mask |= panoptic_seg == int(segment["id"])
            return person_mask if np.any(person_mask) else None

        return None

    person_mask = np.asarray(prediction, dtype=np.int32) == int(person_class_id)
    return person_mask if np.any(person_mask) else None


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
