# **User Guide: YOLO Model Training and Testing**

This guide will walk you through the process of training and testing a YOLO model using the provided Python scripts.

---
## **1. Training the Model**
### **Main Script: `projectDL2025_q1.py` and `projectDL2025_q2.py`**
To train the YOLO model, follow these steps:

1. Open the script `projectDL2025_q1.py` or `projectDL2025_q2.py`.
2. Update the model path to your trained model:

   ```python
   yolo_model = YOLOModel(model_path="best.pt")
   ```

3. Start the training process by running:

   ```python
   yolo_model.train_yolo(resume=False)
   ```

This will initiate the training of the YOLO model using the specified dataset.

---
## **2. Running Tests on Single Images**
### **Main Script: `projectDL2025_q1.py` and `projectDL2025_q2.py`**
To test individual images, follow these steps:

1.  Open the script `projectDL2025_q1.py` or `projectDL2025_q2.py`.
2. Ensure that the test image directory is correctly set:

   ```python
   test_dir = "DataSet/test"
   results_dir = "DataSet/test/results"
   os.makedirs(results_dir, exist_ok=True)
   ```

3. Run the script to process all `.jpg` images in the test directory:

   ```python
   import os

   for test_image in os.listdir(test_dir):
       if test_image.endswith(".jpg"):
           yolo_model.predict_process_bounding_boxes(
               os.path.join(test_dir, test_image),
               os.path.join(results_dir, f"{test_image}.csv"),
               conf_threshold=0.35, iou_threshold=0.45  # Adjusted thresholds
           )
   ```

4. The results will be saved as `.csv` files in the `results` directory.

---
## **3. Batch Testing for Multiple Images**
### **Script: `bounding_box_test.py`**
To test multiple images at once, use `bounding_box_test.py`:

1. Open `bounding_box_test.py` and update the paths:

   ```python
   images_dir = "/Users/administrator/Documents/Python/DeepNeuronalSystems-Final/DataSet/test/images"  # Path to the test images
   output_dir = "/Users/administrator/Documents/Python/DeepNeuronalSystems-Final/DataSet/test/results"  # Path to save the results
   model_path = "best_p2.pt"  # Replace with your trained model path
   ```

2. Run the script to process all images in the test dataset.
3. A summary file will be generated in **Section 2**, containing accuracy results.

---
## **Conclusion**
By following these steps, you can successfully train and test a YOLO model using the provided scripts. If you encounter any issues, ensure the paths are correctly set and that the necessary dependencies are installed.

