import os
import json
import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class YOLOModel:
    def __init__(self, model_path="yolov11n.pt", trained_model_dir="runs/detect/train11/weights"):
        """
        Initializes the YOLO model.
        :param model_path: Path to the pre-trained YOLO model.
        :param trained_model_dir: Directory where trained weights are stored.
        """
        self.model_path = model_path
        self.trained_model_dir = trained_model_dir
        self.trained_model_path = os.path.join(trained_model_dir, "best.pt")
        self.last_model_path = os.path.join(trained_model_dir, "last.pt")

        if os.path.exists(self.last_model_path):
            print(f"Loading last trained model from {self.last_model_path}")
            self.model = YOLO(self.last_model_path)
        elif os.path.exists(self.trained_model_path):
            print(f"Loading best trained model from {self.trained_model_path}")
            self.model = YOLO(self.trained_model_path)
        else:
            print(f"Loading pre-trained model from {self.model_path}")
            self.model = YOLO(self.model_path)

    def convert_labelme_to_yolo(self, json_folder, output_folder):
        """
        Converts LabelMe JSON annotations to YOLO format.
        :param json_folder: Path to the folder containing JSON annotation files.
        :param output_folder: Path where the converted YOLO labels will be saved.
        """
        os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

        for filename in os.listdir(json_folder):
            if filename.endswith(".json"):
                with open(os.path.join(json_folder, filename), 'r') as f:
                    data = json.load(f)
                img_w, img_h = data["imageWidth"], data["imageHeight"]
                txt_filename = os.path.join(output_folder, filename.replace(".json", ".txt"))
                with open(txt_filename, "w") as txt_file:
                    for shape in data["shapes"]:
                        label = 0  # Default class ID
                        (xmin, ymin), (xmax, ymax) = shape["points"]

                        # Ensure values are within valid bounds
                        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), max(0, xmax), max(0, ymax)

                        # Convert to YOLO format (normalized values)
                        x_center = ((xmin + xmax) / 2) / img_w
                        y_center = ((ymin + ymax) / 2) / img_h
                        width = (xmax - xmin) / img_w
                        height = (ymax - ymin) / img_h

                        # Write to file with 6 decimal places
                        txt_file.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def train_yolo(self, data_yaml="data.yaml", epochs=100, batch_size=10, img_size=800, cache="disk", resume=True):
        """
        Trains the YOLO model using the specified dataset.
        :param data_yaml: Path to the dataset configuration file.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param img_size: Image size for training.
        :param cache: Caching strategy ('ram', 'disk', or False).
        :param resume: Whether to resume training from the last checkpoint.
        """
        if resume and os.path.exists(self.last_model_path):
            print("Resuming training...")
            self.model = YOLO(self.last_model_path)

        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            cache=cache,
            resume=resume
        )
        print(f"Training complete. Best model saved at: {self.trained_model_path}")

        # Compare models and keep the best one
        if os.path.exists(self.trained_model_path):
            if os.path.exists(self.last_model_path):
                best_model = YOLO(self.trained_model_path)
                last_model = YOLO(self.last_model_path)
                best_mAP = best_model.val()["metrics/mAP50-95(B)"]
                last_mAP = last_model.val()["metrics/mAP50-95(B)"]
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
        img = cv2.imread(image_path) # Read the image
        results = model.predict(img, save=True, conf=conf_threshold, iou=iou_threshold, augment=use_tta)

        data = []
        for result in results:
            image_name = os.path.basename(image_path)
            for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                xmin, ymin, xmax, ymax = map(int, box)

                # Ensure values are within valid bounds
                xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), max(0, xmax), max(0, ymax)

                data.append([image_name, i + 1, xmin, ymin, xmax, ymax, -1])  # -1 for IOU placeholder

        df = pd.DataFrame(data, columns=["image_name", "scroll_number", "xmin", "ymin", "xmax", "ymax", "iou"])
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure output folder exists
        df.to_csv(output_csv, index=False) # Save results to CSV

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
