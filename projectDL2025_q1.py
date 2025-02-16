import os
import json
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import train_test_split


class YOLOModel:
    def __init__(self, model_path="yolo11n.pt", trained_model_path="yolo11n_trained.pt"):
        """
        Initializes the YOLO model.
        :param model_path: Path to the pre-trained YOLO model.
        :param trained_model_path: Path where the trained model will be saved.
        """
        self.model_path = model_path
        self.trained_model_path = trained_model_path
        self.model = YOLO(self.model_path)  # Load the pre-trained model

    def train_yolo(self, data_yaml="data.yaml", epochs=100, batch_size=16, img_size=640, cache="disk"):
        """
        Trains the YOLO model using the specified dataset.
        :param data_yaml: Path to the dataset configuration file.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param img_size: Image size for training (default: 640x640).
        :param cache: Caching strategy ('ram' for speed, 'disk' for determinism, False for no cache)
        """
        print(f"Training YOLO model with: epochs={epochs}, batch={batch_size}, imgsz={img_size}, cache={cache}")
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            cache=cache,
            optimizer="auto",
            device=None  # Use CPU as in your previous training
        )
        self.model.save(self.trained_model_path)  # Save the trained model

    def predict_process_bounding_boxes(self, image_path: str, output_csv: str) -> None:
        """
        Runs inference on an image and processes bounding boxes into a CSV file.
        :param image_path: Path to the image file.
        :param output_csv: Path to save the results in CSV format.
        """
        model = YOLO(self.trained_model_path)  # Load the trained model
        img = cv2.imread(image_path)  # Read the image
        results = model.predict(img, save=True, conf=0.5)  # Perform object detection

        data = []
        for result in results:
            image_name = os.path.basename(image_path)
            for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                xmin, ymin, xmax, ymax = map(int, box)
                data.append([image_name, i + 1, xmin, ymin, xmax, ymax, -1])  # -1 for IOU placeholder

        df = pd.DataFrame(data, columns=["image_name", "scroll_number", "xmin", "ymin", "xmax", "ymax", "iou"])
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure output folder exists
        df.to_csv(output_csv, index=False)  # Save results to CSV


if __name__ == "__main__":
    """
    Main script execution: trains YOLO and runs predictions on test images.
    """
    yolo_model = YOLOModel()

    # Train the YOLO model with caching enabled
    yolo_model.train_yolo()

    # Run predictions on test images
    for test_image in os.listdir("DataSet/test"):
        if test_image.endswith(".jpg"):
            yolo_model.predict_process_bounding_boxes(
                os.path.join("DataSet/test", test_image),
                f"DataSet/test/results/{test_image}.csv"
            )
