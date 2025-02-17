import os

import cv2
import numpy as np
from dotenv import load_dotenv

import supervisely as sly

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


def color_map(img: np.ndarray, data: np.ndarray, origin: sly.PointLocation) -> np.ndarray:
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    x, y = origin.col, origin.row
    h, w = data.shape[:2]
    mask[y : y + h, x : x + w] = data
    cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    BG_COLOR = np.array([128, 0, 0], dtype=np.uint8)
    mask = np.where(mask == BG_COLOR, 0, mask)
    return mask


def process_image(img_path: str, ann_path: str, project_meta: sly.ProjectMeta) -> np.ndarray:
    img = cv2.imread(img_path)
    ann = sly.Annotation.load_json_file(ann_path, project_meta)
    temp = img.copy()
    for label in ann.labels[::-1]:
        if isinstance(label.geometry, sly.AlphaMask):
            mask = color_map(img, label.geometry.data, label.geometry.origin)
            temp = np.where(np.any(mask > 0, axis=-1, keepdims=True), mask, temp)
    result = cv2.addWeighted(img, 0.5, temp, 0.5, 0).astype(np.uint8)
    return result


def process_dataset(dataset_fs: sly.Dataset, project_meta: sly.ProjectMeta, output_dir: str):
    for item_name in dataset_fs.get_items_names():
        img_path = dataset_fs.get_img_path(item_name)
        ann_path = dataset_fs.get_ann_path(item_name)
        result = process_image(img_path, ann_path, project_meta)
        success = cv2.imwrite(os.path.join(output_dir, item_name), result)
        print(f"Processed {item_name}: {'Success' if success else 'Failed'}")


if __name__ == "__main__":
    project_id = sly.env.project_id()
    api = sly.Api.from_env()

    converted_dir = "./heatmaps"
    sly.fs.mkdir(converted_dir)

    project_path = "./project"
    if not sly.fs.dir_exists(project_path):
        sly.download_project(api, project_id, project_path)

    project_fs = sly.Project(project_path, sly.OpenMode.READ)
    for dataset_fs in project_fs:
        process_dataset(dataset_fs, project_fs.meta, converted_dir)
