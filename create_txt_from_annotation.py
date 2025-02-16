import json
import glob
import os
import cv2


class LabelMeToYOLOConverter:
    def __init__(self, dataset_path):
        """
        Initializes the converter with dataset paths.

        Parameters:
        - dataset_path: Root dataset directory.
        """
        self.dataset_path = dataset_path

        # Define dataset subdirectories
        self.annotations_path = os.path.join(dataset_path, "train/annotations")
        self.yolo_labels_path = os.path.join(dataset_path, "train/labels")
        self.image_path = os.path.join(dataset_path, "train/images")

        # Ensure output directories exist
        os.makedirs(self.yolo_labels_path, exist_ok=True)

    def convert_labelme_to_yolo(self, json_path, output_path, image_size):
        """
        Converts a single LabelMe JSON annotation file to YOLO format.

        Parameters:
        - json_path: Path to the LabelMe JSON file.
        - output_path: Path to save the converted YOLO label file.
        - image_size: Tuple (width, height) of the image
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        w, h = image_size  # Image dimensions
        yolo_annotations = []

        for shape in data["shapes"]:
            label = shape["label"]  # Class label (default: 0)
            points = shape["points"]

            # Get bounding box coordinates
            x_min = min(p[0] for p in points)
            y_min = min(p[1] for p in points)
            x_max = max(p[0] for p in points)
            y_max = max(p[1] for p in points)

            # Convert to YOLO format (normalized values)
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h

            # Ensure values stay within range [0,1]
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)

            # Assuming a single class (change "0" if you have multiple classes)
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save YOLO labels
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_annotations))

    def convert_all_annotations(self):
        """
        Converts all LabelMe JSON annotations in the dataset to YOLO format.
        """
        json_files = glob.glob(os.path.join(self.annotations_path, "*.json"))

        if not json_files:
            print("⚠️ No JSON annotation files found!")
            return

        for json_file in json_files:
            base_name = os.path.basename(json_file).replace(".json", "")
            output_file = os.path.join(self.yolo_labels_path, f"{base_name}.txt")
            image_file = os.path.join(self.image_path, f"{base_name}.jpg")

            if not os.path.exists(image_file):
                image_file = os.path.join(self.image_path, f"{base_name}.png")
                if not os.path.exists(image_file):
                    print(f"⚠️ Missing image for {json_file}, skipping...")
                    continue

            # Get image size dynamically
            image = cv2.imread(image_file)
            if image is None:
                print(f"❌ Error loading image {image_file}, skipping...")
                continue
            height, width, _ = image.shape

            self.convert_labelme_to_yolo(json_file, output_file, (width, height))

        print("✅ LabelMe annotations successfully converted to YOLO format!")

    def verify_dataset(self):
        """
        Checks if all images have corresponding labels and vice versa.
        """
        image_files = {os.path.splitext(f)[0] for f in os.listdir(self.image_path) if
                       f.endswith(('.jpg', '.png', '.jpeg'))}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(self.yolo_labels_path) if f.endswith('.txt')}

        missing_labels = image_files - label_files
        extra_labels = label_files - image_files

        if missing_labels:
            print(f"⚠️ Warning: {len(missing_labels)} images have no corresponding labels!")
            print(missing_labels)

        if extra_labels:
            print(f"⚠️ Warning: {len(extra_labels)} label files have no corresponding images!")
            print(extra_labels)

    def clear_yolo_cache(self):
        """
        Removes the YOLO labels cache to avoid outdated data.
        """
        cache_path = os.path.join(self.yolo_labels_path, "labels.cache")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("🗑️ YOLO cache cleared.")

    def run(self):
        """
        Main function to execute the full conversion and verification process.
        """
        self.convert_all_annotations()
        self.verify_dataset()
        # self.clear_yolo_cache()
        print("🚀 Dataset is ready for training!")


# ---- MAIN ----
if __name__ == "__main__":
    dataset_path = "./DataSet"
    converter = LabelMeToYOLOConverter(dataset_path)
    converter.run()