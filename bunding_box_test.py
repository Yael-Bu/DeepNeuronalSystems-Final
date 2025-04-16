import os
import cv2
import pandas as pd
from ultralytics import YOLO
import json

import os
import pandas as pd


def merge_csv_files(input_dir: str, output_csv: str):
    """
    Merges all CSV files in the specified directory into one CSV file.

    Args:
    input_dir (str): Path to the directory containing the CSV files.
    output_csv (str): Path to the output CSV file where the merged results will be saved.
    """
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('_results_iou.csv')]

    # Initialize an empty list to store dataframes
    all_data = []

    # Loop through all CSV files and append them to the list
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)
        all_data.append(df)

    # Concatenate all dataframes into one
    merged_data = pd.concat(all_data, ignore_index=True)

    # Save the merged dataframe to a new CSV file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged_data.to_csv(output_csv, index=False)
    print(f"All results merged into {output_csv}")


def load_model(model_path: str):
    """
    Loads the YOLO model from the specified path.

    Args:
    model_path (str): Path to the model weights file (best_p2.pt).

    Returns:
    YOLO model: Loaded YOLOv5 model.
    """
    model = YOLO(model_path)  # Load the YOLO model with the given path
    return model

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
    box1 (list): Coordinates of the first bounding box [xmin, ymin, xmax, ymax].
    box2 (list): Coordinates of the second bounding box [xmin, ymin, xmax, ymax].

    Returns:
    float: IoU value between 0 and 1.
    """
    # Get the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def sort_boxes(boxes):
    """
    Sort boxes based on their xmin (left side of the box).

    Args:
    boxes (list): List of bounding boxes as [xmin, ymin, xmax, ymax]

    Returns:
    list: Sorted list of boxes.
    """
    return sorted(boxes, key=lambda x: x[0])  # Sort by xmin

def predict_process_bounding_boxes(image_path: str, output_csv: str, model_path: str, conf_threshold: float = 0.4,
                                   iou_threshold: float = 0.5):
    """
    Processes an image to detect bounding boxes around scroll segments and calculates IOU with ground truth.
    Saves the bounding box data to a CSV file.

    Args:
    image_path (str): Path to the input image.
    output_csv (str): Path to the output CSV file.
    model_path (str): Path to the trained model.
    conf_threshold (float): Confidence threshold for predictions.
    iou_threshold (float): IOU threshold for considering a detection as valid.
    """
    # Load model
    model = YOLO(model_path)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Failed to read image: {image_path}")
    print(f"Read image {image_path}")

    # Predict using the model
    results = model.predict(img, conf=conf_threshold, iou=iou_threshold)

    data = []  # To store the bounding box data

    # Load ground truth bounding boxes (from JSON)
    json_path = image_path.replace('images', 'annotations').replace('.jpg', '.json').replace('.png', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            ground_truth = json.load(file)["shapes"]
        ground_truth = [
            [int(shape["points"][0][0]), int(shape["points"][0][1]),
             int(shape["points"][1][0]), int(shape["points"][1][1])]  # Convert points to xmin, ymin, xmax, ymax
            for shape in ground_truth if len(shape["points"]) == 2
        ]
    else:
        ground_truth = []

    # Sort the predicted boxes and ground truth boxes
    predicted_boxes_sorted = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            xmin, ymin, xmax, ymax = map(int, box)
            predicted_box = [xmin, ymin, xmax, ymax]
            predicted_boxes_sorted.append(predicted_box)

    # Sort the ground truth boxes based on xmin
    ground_truth_sorted = []
    for gt_box in ground_truth:
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
        ground_truth_sorted.append([gt_xmin, gt_ymin, gt_xmax, gt_ymax])

    # Sort both predicted and ground truth boxes by xmin
    predicted_boxes_sorted = sort_boxes(predicted_boxes_sorted)
    ground_truth_sorted = sort_boxes(ground_truth_sorted)
    print(f"predicted_boxes_sorted: {predicted_boxes_sorted}")
    print(f"ground_truth_sorted: {ground_truth_sorted}")


    # Process each predicted bounding box and calculate IOU
    for i, predicted_box in enumerate(predicted_boxes_sorted):
        image_name = os.path.basename(image_path)
        xmin, ymin, xmax, ymax = predicted_box

        # Convert predicted bounding box to relative coordinates (0 to 1 range)
        height, width = img.shape[:2]
        relative_predicted_box = [
            xmin / width, ymin / height, xmax / width, ymax / height
        ]

        # Normalize ground truth bounding boxes to [0, 1] range
        ground_truth_normalized = []
        for gt_box in ground_truth_sorted:
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
            normalized_gt_box = [
                gt_xmin / width, gt_ymin / height, gt_xmax / width, gt_ymax / height
            ]
            ground_truth_normalized.append(normalized_gt_box)

        # Calculate IOU
        iou = -1  # Default IOU value if no valid overlap with ground truth
        for gt_box in ground_truth_normalized:
            iou_value = calculate_iou(relative_predicted_box, gt_box)

            # Use maximum IOU with ground truth
            if iou_value > iou:
                iou = iou_value

        # Append the data for the CSV (image_name, scroll_number, bounding box coordinates, and IOU)
        data.append([image_name, i + 1, relative_predicted_box[0], relative_predicted_box[1],
                     relative_predicted_box[2], relative_predicted_box[3], iou])

    # Save the results to CSV
    if data:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df = pd.DataFrame(data, columns=["image_name", "scroll_number", "xmin", "ymin", "xmax", "ymax", "iou"])
        df.to_csv(output_csv, index=False)

def process_detailed_bounding_boxes(image_path: str, output_csv: str, model_path: str, conf_threshold: float = 0.4,
                                    iou_threshold: float = 0.5) -> None:
    """
    Processes an image to detect detailed bounding boxes for both large and small scroll segments.
    Saves the bounding box data to a CSV file.

    Args:
    image_path (str): Path to the input image.
    output_csv (str): Path to the output CSV file.
    model_path (str): Path to the trained model.
    conf_threshold (float): Confidence threshold for predictions.
    iou_threshold (float): IOU threshold for considering a detection as valid.
    """
    # Load model
    model = YOLO(model_path)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Failed to read image: {image_path}")
    print(f"Read image {image_path}")

    # Predict using the model
    results = model.predict(img, conf=conf_threshold, iou=iou_threshold)

    data = []  # To store the bounding box data

    # Load ground truth bounding boxes (from JSON)
    json_path = image_path.replace('images', 'annotations').replace('.jpg', '.json').replace('.png', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            ground_truth = json.load(file)["shapes"]
        ground_truth = [
            [int(shape["points"][0][0]), int(shape["points"][0][1]),
             int(shape["points"][1][0]), int(shape["points"][1][1])]  # Convert points to xmin, ymin, xmax, ymax
            for shape in ground_truth if len(shape["points"]) == 2
        ]
    else:
        ground_truth = []

    # Sort the predicted boxes and ground truth boxes
    predicted_boxes_sorted = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            xmin, ymin, xmax, ymax = map(int, box)
            predicted_box = [xmin, ymin, xmax, ymax]
            predicted_boxes_sorted.append(predicted_box)

    # Sort the ground truth boxes based on xmin
    ground_truth_sorted = []
    for gt_box in ground_truth:
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
        ground_truth_sorted.append([gt_xmin, gt_ymin, gt_xmax, gt_ymax])

    # Sort both predicted and ground truth boxes by xmin
    predicted_boxes_sorted = sort_boxes(predicted_boxes_sorted)
    ground_truth_sorted = sort_boxes(ground_truth_sorted)

    # Process each predicted bounding box and calculate IOU
    for i, predicted_box in enumerate(predicted_boxes_sorted):
        image_name = os.path.basename(image_path)
        xmin, ymin, xmax, ymax = predicted_box

        # Convert predicted bounding box to relative coordinates (0 to 1 range)
        height, width = img.shape[:2]
        relative_predicted_box = [
            xmin / width, ymin / height, xmax / width, ymax / height
        ]

        # Normalize ground truth bounding boxes to [0, 1] range
        ground_truth_normalized = []
        for gt_box in ground_truth_sorted:
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
            normalized_gt_box = [
                gt_xmin / width, gt_ymin / height, gt_xmax / width, gt_ymax / height
            ]
            ground_truth_normalized.append(normalized_gt_box)

        # Calculate IOU
        iou = -1  # Default IOU value if no valid overlap with ground truth
        for gt_box in ground_truth_normalized:
            iou_value = calculate_iou(relative_predicted_box, gt_box)

            # Use maximum IOU with ground truth
            if iou_value > iou:
                iou = iou_value

        # Assign a scroll number based on the location (use `scroll_number` to count the segment)
        scroll_number = i + 1

        # Append the data for the CSV (image_name, scroll_number, bounding box coordinates, and IOU)
        data.append([image_name, scroll_number, relative_predicted_box[0], relative_predicted_box[1],
                     relative_predicted_box[2], relative_predicted_box[3], iou])

    # Save the results to CSV
    if data:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df = pd.DataFrame(data, columns=["image_name", "scroll_number", "xmin", "ymin", "xmax", "ymax", "iou"])
        df.to_csv(output_csv, index=False)


def process_all_images_in_directory(images_dir: str, output_dir: str, model_path: str, file_name: str = "trained_results.csv") -> None:
    """
    Processes all images in the specified directory and saves the bounding box results to CSV.

    Args:
    images_dir (str): Path to the directory containing training images.
    output_dir (str): Path to save the results CSV files.
    model_path (str): Path to the trained model.
    """
    # Process each image in the directory
    for image_name in os.listdir(images_dir):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(images_dir, image_name)
            output_csv_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_results_iou.csv")
            predict_process_bounding_boxes(image_path, output_csv_path, model_path)

    merge_csv_files(output_dir, os.path.join(output_dir, file_name))


def main():
    """
    Main function to process all images in the training directory and save results to CSV files.
    """
    images_dir = "/Users/administrator/Documents/Python/DeepNeuronalSystems-Final/DataSet/test/images"  # Path to the train images
    output_dir = "/Users/administrator/Documents/Python/DeepNeuronalSystems-Final/DataSet/test/results"  # Path to save the results
    model_path = "best_p2.pt"  # Replace with your trained model path

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all images in the specified directory
    #process_all_images_in_directory(images_dir, output_dir, model_path, file_name="test_results")

    one_image_path = "/Users/administrator/Documents/Python/DeepNeuronalSystems-Final/DataSet/test/images/M42934-1-E.jpg"
    predict_process_bounding_boxes(one_image_path, "/Users/administrator/Documents/Python/DeepNeuronalSystems-Final/DataSet/test/results/test.csv", model_path = "best.pt")

if __name__ == "__main__":
    main()
