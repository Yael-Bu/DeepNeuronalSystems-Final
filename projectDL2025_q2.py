import os
import shutil
import cv2
import yaml
import pandas as pd
from ultralytics import YOLO
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

# EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01, metric='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float('inf') if metric == 'loss' else -float('inf')
        self.counter = 0
        self.early_stop = False
        self.metric = metric

    def __call__(self, val_metric):
        improvement = (val_metric < self.best_metric - self.min_delta) if self.metric == 'loss' else (
                    val_metric > self.best_metric + self.min_delta)

        if improvement:
            self.best_metric = val_metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True


class YOLOModel:
    def __init__(self, model_path="yolov11s.pt", train_folder="runs/detect/train66"):
        self.model_path = model_path
        self.trained_model_path = os.path.join(train_folder, "weights/best.pt")
        self.last_model_path = os.path.join(train_folder, "weights/last.pt")
        self.model = None

    def load_model(self):
        if os.path.exists(self.trained_model_path):
            self.model = YOLO(self.trained_model_path)
        elif os.path.exists(self.last_model_path):
            self.model = YOLO(self.last_model_path)
        else:
            self.model = YOLO(self.model_path)

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

        # Copy the images to the train directory
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

    def train_yolo(self, train_dir="DataSet/train/images", data_yaml="data.yaml", max_epochs=50, batch_size=10,
                   img_size=800, cache="disk", step_epochs=5):
        train_images = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if
                        fname.endswith(('.jpg', '.png'))]

        self.load_model()

        # Split data for training and validation (remove K-fold)
        train_data = train_images[:int(len(train_images) * 0.8)]  # 80% for training
        val_data = train_images[int(len(train_images) * 0.8):]  # 20% for validation

        # Split images and annotations for each set
        self.split_images_and_annotations(train_data, val_data, data_yaml)

        # Define early stopping
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, metric='mAP')

        # Create optimizer manually
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

        total_epochs = 0

        while total_epochs < max_epochs:
            remaining_epochs = min(step_epochs, max_epochs - total_epochs)  # לא לרוץ מעבר ל-max_epochs

            train_metrics = self.model.train(
                data=data_yaml,
                epochs=remaining_epochs,  # להריץ כל פעם קבוצת epochs קטנה
                batch=batch_size,
                imgsz=img_size,
                cache=cache,
                resume=False,  # להמשיך מאיפה שנשארנו
                optimizer='auto',
                lr0=0.0003,
                lrf=0.05,
                momentum=0.9,
                weight_decay=1e-4,
                augment=True,
                verbose=True
            )

            val_mAP = train_metrics.box.map  # הוצאת המטריקות
            print(f"Validation mAP: {val_mAP}")

            scheduler.step(val_mAP)
            early_stopping(val_mAP)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {total_epochs + remaining_epochs}. Training stopped.")
                break

            total_epochs += remaining_epochs

        print(f"Training finished after {total_epochs} epochs!")


def process_detailed_bounding_boxes(image_paths: list[str], output_csv: str, model_path = "best_p2.pt",  conf_threshold=0.4, iou_threshold=0.35) -> list[str]:
    """
    Processes a list of image file paths to detect detailed bounding boxes
    for both large and small scroll segments.
    Saves the bounding box data to a CSV file.

    Args:
        image_paths (list[str]): List of full paths to the input images.
        output_csv (str): Path to the output CSV file.
        conf_threshold
        iou_threshold

    Returns:
        list[str]: List of full paths (path + file name) of processed images.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model not found! Train the model first.")

    model = YOLO(model_path)
    processed_images = []
    data = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"❌ Failed to read image: {image_path}")

        results = model.predict(img, conf=conf_threshold, iou=iou_threshold)
        image_name = os.path.basename(image_path)

        # Load ground truth bounding boxes from JSON
        json_path = image_path.replace('images', 'annotations').replace('.jpg', '.json').replace('.png', '.json')
        ground_truth_boxes = []
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                ground_truth = json.load(file).get("shapes", [])
            ground_truth_boxes = [
                [int(shape["points"][0][0]), int(shape["points"][0][1]),
                 int(shape["points"][1][0]), int(shape["points"][1][1])]
                for shape in ground_truth if len(shape["points"]) == 2
            ]

        predicted_boxes_sorted = sorted([list(map(int, box)) for result in results for box in result.boxes.xyxy.cpu().numpy()], key=lambda box: box[0])

        # Process bounding boxes
        for i, predicted_box in enumerate(predicted_boxes_sorted):
            xmin, ymin, xmax, ymax = predicted_box
            data.append([image_name, i + 1, xmin, ymin, xmax, ymax])

        # Add ground truth bounding boxes
        #for i, gt_box in enumerate(ground_truth_boxes):
            #xmin, ymin, xmax, ymax = gt_box
            #data.append([image_name, f"GT-{i + 1}", xmin, ymin, xmax, ymax])

        processed_images.append(image_path)

    if data:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df = pd.DataFrame(data, columns=["image_name", "scroll_number", "xmin", "ymin", "xmax", "ymax"])
        df.to_csv(output_csv, index=False)

    return processed_images


if __name__ == "__main__":
    #yolo_model = YOLOModel()
   # yolo_model.train_yolo()

    test_dir = "DataSet/train/images"
    results_dir = "DataSet/train/results"
    os.makedirs(results_dir, exist_ok=True)

    image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith(".jpg")]
    output_csv = os.path.join(results_dir, "training_bounding_boxes.csv")

    if image_paths:
        images = process_detailed_bounding_boxes(image_paths, output_csv, conf_threshold=0.4, iou_threshold=0.35)
        print(f"images are: {images}")

