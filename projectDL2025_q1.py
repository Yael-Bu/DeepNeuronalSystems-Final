import os
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch.optim as optim
from sklearn.model_selection import KFold

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.2, metric='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = -float('inf') if metric != 'loss' else float('inf')
        self.counter = 0
        self.early_stop = False
        self.metric = metric  # metric could be 'loss', 'mAP', 'F1', etc.

    def __call__(self, val_metric):
        if self.metric == 'loss':
            if val_metric < self.best_metric - self.min_delta:
                self.best_metric = val_metric
                self.counter = 0
            else:
                self.counter += 1
        else:  # mAP, F1 or others
            if val_metric > self.best_metric + self.min_delta:
                self.best_metric = val_metric
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

class YOLOModel:
    def __init__(self, model_path="yolov11n.pt", trained_model_dir="runs/detect/train29/weights"):
        self.model_path = model_path
        self.trained_model_dir = trained_model_dir
        self.trained_model_path = os.path.join(trained_model_dir, "best.pt")
        self.last_model_path = os.path.join(trained_model_dir, "last.pt")

        # בודק איזה מודל טוב יותר (אם שניהם קיימים)
        if os.path.exists(self.trained_model_path) and os.path.exists(self.last_model_path):
            best_model = YOLO(self.trained_model_path)
            last_model = YOLO(self.last_model_path)
            if best_model.val().box.map > last_model.val().box.map:
                print(f"Loading best model from {self.trained_model_path}")
                self.model = best_model
            else:
                print(f"Loading last trained model from {self.last_model_path}")
                self.model = last_model
        elif os.path.exists(self.last_model_path):
            print(f"Loading last trained model from {self.last_model_path}")
            self.model = YOLO(self.last_model_path)
        elif os.path.exists(self.trained_model_path):
            print(f"Loading best trained model from {self.trained_model_path}")
            self.model = YOLO(self.trained_model_path)
        else:
            print(f"Loading pre-trained model from {model_path}")
            self.model = YOLO(model_path)

    def train_yolo(self, data_yaml="data.yaml", epochs=10, batch_size=10, img_size=800, cache="disk", resume=True):
        if resume and os.path.exists(self.last_model_path):
            print("Resuming training...")
            self.model = YOLO(self.last_model_path)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5, min_lr=1e-6,
                                                         verbose=True)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(epochs))):
            print(f"Training fold {fold + 1}/{kfold.get_n_splits()}")

            early_stopping = EarlyStopping(patience=5, min_delta=0.01, metric='mAP')

            self.model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                cache=cache,
                resume=resume,
                optimizer="Adam"  # שימוש באדם
            )

            val_metrics = self.model.val()
            mAP50_95 = val_metrics.box.map

            print(f"Validation mAP50-95: {mAP50_95}")
            scheduler.step(mAP50_95)  # עדכון LR
            early_stopping(mAP50_95)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at fold {fold + 1}. Stopping training.")
                break

        print(f"Training complete. Best model saved at: {self.trained_model_path}")

        # Compare models and keep the best one
        if os.path.exists(self.trained_model_path):
            if os.path.exists(self.last_model_path):
                best_model = YOLO(self.trained_model_path)
                last_model = YOLO(self.last_model_path)
                best_mAP = best_model.val().box.map
                last_mAP = last_model.val().box.map
                if best_mAP > last_mAP:
                    os.rename(self.trained_model_path, self.last_model_path)
                    print("Updated last.pt with better model.")
                else:
                    print("Keeping previous last.pt as it has better performance.")
            else:
                os.rename(self.trained_model_path, self.last_model_path)
                print("Saved best model as last.pt")

    def predict_process_bounding_boxes(self, image_path: str, output_csv: str, conf_threshold=0.4, iou_threshold=0.5, use_tta=True):
        """
         Runs inference on an image and processes bounding boxes into a CSV file.
         :param image_path: Path to the image file.
         :param output_csv: Path to save the results in CSV format.
         """
        if not os.path.exists(self.last_model_path):
            raise FileNotFoundError("Train the model first!")

        model = YOLO(self.last_model_path)

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image {image_path}")

        results = model.predict(img, save=True, conf=conf_threshold, iou=iou_threshold, augment=use_tta)

        data = []
        h, w, _ = img.shape  # Get image dimensions
        for result in results:
            image_name = os.path.basename(image_path)
            for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                xmin, ymin, xmax, ymax = map(int, box)

                # Ensure values are within valid bounds
                xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), max(0, xmax), max(0, ymax)

                data.append([image_name, i + 1, xmin, ymin, xmax, ymax, -1])  # -1 for IOU placeholder

        if data:
            df = pd.DataFrame(data, columns=["image_name", "scroll_number", "xmin", "ymin", "xmax", "ymax", "iou"])
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)

    def optimize_model(self, quantize=False, convert_trt=False):
        """Optimizes the model for faster inference."""
        if quantize:
            print("Applying INT8 quantization...")
            self.model.export(format='onnx', int8=True)
        if convert_trt:
            print("Converting to TensorRT...")
            self.model.export(format='engine')


if __name__ == "__main__":
    """
    Main script execution: converts annotations, trains YOLO, and runs predictions on test images.
    """
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
                conf_threshold=0.35, iou_threshold=0.45  # Adjusted thresholds
            )