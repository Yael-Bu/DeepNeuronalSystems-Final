
# YOLO Model Training and Testing Guide

This guide provides instructions for training and testing a YOLO model using the provided Python scripts.

---

## **1. Training the Model**
### **Main Script: `projectDL2025_q1.py` and `projectDL2025_q2.py`**

To train the YOLO model, follow these steps:

1. Open the script `projectDL2025_q1.py` or `projectDL2025_q2.py`.
2. Update the model path to your trained model:

   ```python
   yolo_model = YOLOModel(model_path="best.pt")
Start the training process by running:
yolo_model.train_yolo(resume=False)
This will initiate the training of the YOLO model using the specified dataset.

2. Running Tests on Single Images

Main Script: projectDL2025_q1.py and projectDL2025_q2.py
To test individual images, follow these steps:

Open the script projectDL2025_q1.py or projectDL2025_q2.py.
Ensure that the test image directory is correctly set:
test_dir = "DataSet/test"
results_dir = "DataSet/test/results"
os.makedirs(results_dir, exist_ok=True)
Run the script to process all .jpg images in the test directory:
import os

for test_image in os.listdir(test_dir):
    if test_image.endswith(".jpg"):
        yolo_model.predict_process_bounding_boxes(
            os.path.join(test_dir, test_image),
            os.path.join(results_dir, f"{test_image}.csv"),
            conf_threshold=0.35, iou_threshold=0.45  # Adjusted thresholds
        )
The results will be saved as .csv files in the results directory.
3. Batch Testing for Multiple Images

Script: bounding_box_test.py
To test multiple images at once, use bounding_box_test.py:

Open bounding_box_test.py and update the paths:
images_dir = "/Users/administrator/Documents/Python/DeepNeuronalSystems-Final/DataSet/test/images"  # Path to the test images
output_dir = "/Users/administrator/Documents/Python/DeepNeuronalSystems-Final/DataSet/test/results"  # Path to save the results
model_path = "best_p2.pt"  # Replace with your trained model path
Run the script to process all images in the test dataset.
A summary file will be generated in Section 2, containing accuracy results.
4. Running the Bounding Box Prediction

Main Script: projectDL2025_q1.py
To run the predict_process_bounding_boxes function, follow these steps:

Set the directories: Ensure that the paths to the image directory and results directory are correctly set in your script:
test_dir = "DataSet/test"  # Path to the images you want to process
results_dir = "DataSet/test/results"  # Path where the results will be saved
os.makedirs(results_dir, exist_ok=True)  # Create the results directory if it doesn't exist
Define the YOLO model object:
yolo_model = YOLOModel()
Run the function on individual images: The following code will iterate through all .jpg images in the test_dir and process them with the predict_process_bounding_boxes function:
for test_image in os.listdir(test_dir):
    if test_image.endswith(".jpg"):
        yolo_model.predict_process_bounding_boxes(
            os.path.join(test_dir, test_image),
            os.path.join(results_dir, f"{test_image}.csv"),
            conf_threshold=0.35, iou_threshold=0.45  # Adjusted thresholds
        )
Results saved in CSV: The results for each image will be saved as a .csv file in the results_dir, where each file contains the predicted bounding boxes.
Function Parameters
predict_process_bounding_boxes(image_path, output_csv, conf_threshold=0.4, iou_threshold=0.5, use_tta=True)
Parameters:
image_path: Path to the image file that you want to process.
output_csv: The path where the bounding box results will be saved as a CSV file.
conf_threshold: The confidence threshold for detecting bounding boxes. Default value is 0.4. You can adjust this to filter detections based on confidence.
iou_threshold: The IoU (Intersection over Union) threshold for non-maximum suppression (NMS). Default value is 0.5. This helps filter overlapping bounding boxes.
use_tta: If set to True, Test Time Augmentation (TTA) will be applied to improve predictions. Default is True.
Returns:
The function does not return any value directly. It saves the processed bounding box data to the specified CSV file.
Example Usage
To run the bounding box prediction on your dataset, ensure that the test_dir and results_dir paths are set correctly, and then execute the script.

# Example:
yolo_model = YOLOModel()
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
5. Running the Bounding Box Detection

Main Script: projectDL2025_q2.py
To run the process_detailed_bounding_boxes function, follow these steps:

Set the directories: Ensure that the paths to the image directory and results directory are correctly set in your script:
test_dir = "DataSet/train/images"  # Path to the images you want to process
results_dir = "DataSet/train/results"  # Path where the results will be saved
os.makedirs(results_dir, exist_ok=True)  # Create the results directory if it doesn't exist
Prepare the list of images: The function will process all .jpg images in the test_dir directory. The list of images will be automatically created:
image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith(".jpg")]
Set the output CSV path: Define the path where the results (bounding boxes) will be saved as a CSV file:
output_csv = os.path.join(results_dir, "training_bounding_boxes.csv")
Run the function: If the images are found in the image_paths list, the following code will call the process_detailed_bounding_boxes function to process the images:
if image_paths:
    images = process_detailed_bounding_boxes(image_paths, output_csv, conf_threshold=0.4, iou_threshold=0.35)
    print(f"Processed images: {images}")
This will process each image in the list, detect bounding boxes, and save the results to the training_bounding_boxes.csv file.
2. Function Parameters

process_detailed_bounding_boxes(image_paths, output_csv, conf_threshold=0.4, iou_threshold=0.35)
Parameters:
image_paths: A list of file paths to the images to be processed. You can generate this list by scanning your image directory.
output_csv: The path to the CSV file where the bounding box results will be stored.
model_path: trained model path.
conf_threshold: The confidence threshold for detecting bounding boxes. Default value is 0.4. You can adjust this to filter detections based on confidence.
iou_threshold: The IoU (Intersection over Union) threshold for non-maximum suppression (NMS). Default value is 0.35. This value helps filter out overlapping bounding boxes.
Returns:
The function returns a list of processed images along with their respective bounding box data.
3. Example Usage

To run the bounding box detection on your dataset, ensure that the test_dir and results_dir paths are set correctly, and then execute the script.

# Example:
image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith(".jpg")]
output_csv = os.path.join(results_dir, "training_bounding_boxes.csv")

if image_paths:
    images = process_detailed_bounding_boxes(image_paths, output_csv, conf_threshold=0.4, iou_threshold=0.35)
    print(f"Processed images: {images}")
This will process all images in the test_dir, detect bounding boxes, and save the results to training_bounding_boxes.csv in the results_dir.

Conclusion

By following these steps, you can successfully train and test a YOLO model using the provided scripts. If you encounter any issues, ensure the paths are correctly set and that the necessary dependencies are installed.