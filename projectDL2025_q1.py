import os
import shutil
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch.optim as optim
from sklearn.model_selection import KFold
import yaml


# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.2, metric='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = -float('inf') if metric != 'loss' else float('inf')
        self.counter = 0
        self.early_stop = False
        self.metric = metric

    def __call__(self, val_metric):
        if self.metric == 'loss':
            if val_metric < self.best_metric - self.min_delta:
                self.best_metric = val_metric
                self.counter = 0
            else:
                self.counter += 1
        else:
            if val_metric > self.best_metric + self.min_delta:
                self.best_metric = val_metric
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


class YOLOModel:
    def __init__(self, model_path="yolov11s.pt", trained_model_dir="runs/detect/train43/weights"):
        self.model_path = model_path
        self.trained_model_dir = trained_model_dir
        self.trained_model_path = os.path.join(trained_model_dir, "best.pt")
        self.last_model_path = os.path.join(trained_model_dir, "last.pt")

        if os.path.exists(self.trained_model_path):
            self.model = YOLO(self.trained_model_path)
        elif os.path.exists(self.last_model_path):
            self.model = YOLO(self.last_model_path)
        else:
            self.model = YOLO(model_path)

    def load_yaml(self, yaml_path="data.yaml"):
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def split_images_and_annotations(self, train_data, val_data, data_yaml="data.yaml"):
        """
        Copy images and their corresponding annotation files into train and validation directories.
        :param train_data: List of paths to training images.
        :param val_data: List of paths to validation images.
        :param data_yaml: Path to the YAML file containing dataset configuration.
        """
                # Load YAML configuration
        config = self.load_yaml(data_yaml)
        # Get image paths from the directories defined in the YAML
        train_dir = config['train']
        val_dir =  config['val']
        train_images_dir = os.path.join(train_dir, 'images')
        val_images_dir = os.path.join(val_dir, 'images')
        train_annotations_dir = os.path.join(train_dir, 'labels')  # Assuming 'labels' for annotations
        val_annotations_dir = os.path.join(val_dir, 'labels')


        # Create the directories if they don't exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(train_annotations_dir, exist_ok=True)
        os.makedirs(val_annotations_dir, exist_ok=True)

        # Delete all files in the existing directories (if any)
        for f in os.listdir(train_images_dir):
            file_path = os.path.join(train_images_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for f in os.listdir(val_images_dir):
            file_path = os.path.join(val_images_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Delete annotation files (if any)
        for f in os.listdir(train_annotations_dir):
            file_path = os.path.join(train_annotations_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for f in os.listdir(val_annotations_dir):
            file_path = os.path.join(val_annotations_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # לCopy the images to the train directory
        for image_path in train_data:
            shutil.copy(image_path, os.path.join(train_images_dir, os.path.basename(image_path)))
            annotation_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(annotation_path):
                shutil.copy(annotation_path, os.path.join(train_annotations_dir, os.path.basename(annotation_path)))

        # Copy the images to the validation directory
        for image_path in val_data:
            shutil.copy(image_path, os.path.join(val_images_dir, os.path.basename(image_path)))
            annotation_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(annotation_path):
                shutil.copy(annotation_path, os.path.join(val_annotations_dir, os.path.basename(annotation_path)))

        print(f"Images and annotations have been split into {train_dir}, {train_annotations_dir} and {val_dir}, {val_annotations_dir}.")


    def train_yolo(self, train_dir="DataSet/train/images", data_yaml="data.yaml",epochs=50, batch_size=10, img_size=800, cache="disk", resume=False):
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7,
                                                         verbose=True)

        # Collect image files from the directories (assuming all files are images)
        train_images = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if
                        fname.endswith(('.jpg', '.png'))]
        print(f"train_images = {train_images}")

        dataset_size = len(train_images) # Now the size is based on the actual number of images
        indices = np.arange(dataset_size)

        # Define the augmentation configurations
        augment_config = {
            'mixup': 0.2,  # 20% MixUp image blending
            'mosaic': 1.0,  # Mosaic enabled 100% of the time
            'hsv_h': 0.015,  # Hue adjustment
            'hsv_s': 0.7,  # Saturation adjustment
            'hsv_v': 0.4,  # Value (brightness) adjustment
            'fliplr': 0.5,  # Horizontal flip
            'flipud': 0.5  # Vertical flip
        }

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            print(f"Training fold {fold + 1}/{kfold.get_n_splits()}")
            print(f"train_idx = {train_idx}, val_idx = {val_idx}")

            # Prepare train and validation datasets based on the indices
            train_data = [train_images[i] for i in train_idx]  # Access images in the 'train' section
            val_data = [train_images[i] for i in val_idx]  # Access images in the 'val' section
            self.split_images_and_annotations(train_data, val_data)

            early_stopping = EarlyStopping(patience=5, min_delta=0.01, metric='mAP')

            # Train model with augmentations as hyp (hyperparameters)
            train_metrics = self.model.train(data=data_yaml, epochs=epochs, batch=batch_size, imgsz=img_size, cache=cache,
                             resume=resume, optimizer="AdamW", augment=False, verbose=True)
            print(f"Train Results: {train_metrics.results_dict}")

            # Validate and compute mAP
            val_metrics = self.model.val()
            mAP50_95 = val_metrics.box.map
            print(f"Validation mAP50-95: {mAP50_95}")

            scheduler.step(mAP50_95)
            early_stopping(val_metrics.box.map)

            if early_stopping.early_stop:
                print(f"Early stopping at fold {fold + 1}")
                break

    def predict_process_bounding_boxes(self, image_path, output_csv, conf_threshold=0.4, iou_threshold=0.5,
                                       use_tta=True):
        if not os.path.exists(self.last_model_path):
            raise FileNotFoundError("Train the model first!")

        model = YOLO(self.last_model_path)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image {image_path}")

        results = model.predict(img, save=True, conf=conf_threshold, iou=iou_threshold, augment=use_tta)
        data = []
        h, w, _ = img.shape

        for result in results:
            image_name = os.path.basename(image_path)
            for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                xmin, ymin, xmax, ymax = map(int, box)
                xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), max(0, xmax), max(0, ymax)
                data.append([image_name, i + 1, xmin, ymin, xmax, ymax, -1])

        if data:
            df = pd.DataFrame(data, columns=["image_name", "scroll_number", "xmin", "ymin", "xmax", "ymax", "iou"])
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    yolo_model = YOLOModel()
    yolo_model.train_yolo(resume=False)
    test_dir = "DataSet/test"
    results_dir = "DataSet/test/results"
    os.makedirs(results_dir, exist_ok=True)

    for test_image in os.listdir(test_dir):
        if test_image.endswith(".jpg"):
            yolo_model.predict_process_bounding_boxes(
                os.path.join(test_dir, test_image),
                os.path.join(results_dir, f"{test_image}.csv"),
                conf_threshold=0.4, iou_threshold=0.35
            )
