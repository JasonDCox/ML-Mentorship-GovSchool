# Desktop Operation
## Overview
## Installation
1. All the installation for desktop Tensorflow is covered in Nicholas Renotte's tutorial (linked below)
    - I have provided the same notebook files he provides but with added features that:
        - address issues with formatting imported annotations
        - addresses issues with images being rotated
        - goes through all images and records all annotations for separate evaluation software
        - takes in a video and outputs the video with predictions overlayed 
    - To get these added features, replace his two notebook files with the matching ones provided in this github.
    - The video goes through installing Tensorflow Object Detection along with CUDA and CUDNN, which is much better covered in an in-depth video rather than a github page
2. The installation for the evaluaion software is covered in Rafael Padilla's github (linked below)
    - This should be installed in a completely different folder since the image and detection folders can easily be selected in the software.
3. Copy the provided evaluation_stuff folder into the TFODCourse folder created by following Nicholas Renotte's video.
## Layout
## Steps
### Creating, training, and exporting a model
1. Create/import image dataset with PASCAL VOC annotations.
    1. Creating/annotating custom images is covered in Nicholas Renotte's tutorial
    2. There are custom code segments to help ensure imported annotations are in the correct order/format (it is very specific)
2. Create label map, tfrecords, and update the config file
    1. Change the label's array to match the exact names of your model's classes
    2. There are code segments for all of these and Nicholas Renotte's video covers them in depth
3. Run Training Script
    1. It is recomended to have 2000 training steps per class
    2. It is also recomended to copy the command into terminal to see the progress as it goes
4. Run Evaluation Script and tensorboard
    1. This will ensure the model is up to the performance you want and has fully flattened out (developed)
    2. Both of these are covered in Nicholas Renotte's video (tensorboard command not in notebook)
5. Run the detect on images and video
    1. First run the segments under Load Train Model From Checkpoint
        1. You must update checkpoint number to be the latest in the folder
    2. Next run the segments under the Detect from all Test Images to run through all the images and log the detections for evaluation
        1. Define the save_folder variable with the name of the folder you want to save the resulting images in.
    4. Lastly, run the segments under Making a prediction video if you want to insert detections into a pre-filmed video
6. Export the model
    1. Run the code segment under Freezing the Graph
### Evaluating results for models
1. Everything will be done in the evaluation_stuff folder
2. Copy the xml annotations from the test data into the annotations folder
3. Copy the images from the test data into the images folder
4. For tfLite, go into the exported folder (object detection) from the Jetson, go into the tfLite folder and copy the following into the tfLite folder under evaluation_stuff
    1. tfLite_results.csv
    2. tfLite.mp4
    3. images from tfLite_results into the image_results folder
5. For TensorRT, go into the exported folder (object detection) from the Jetson, go into the TF-TRT folder and copy the following into the tensorRT folder under evaluation_stuff
    1. tensorRT_results.csv
    2. tensorRT.mp4
    3. result_images folder
6. For Tensorflow copy the files from ______ into the tensorFlow folder in evaluation_stuff
    1. Tensorflow.mp4 from the TFODCourse folder (created in tutorial)
    2. Tensorflow_results.csv from the TFODCourse folder (created in tutorial)
    3. Tensorflow's resulting images from the save folder you specify in the code segment that runs through all the images
7. Double check that every model folder (Tensorflow, tfLite, and TensorRT) has a results csv along with a video and result images (if wanted)
8. Open the custom evaluation notebook with the tfod environment
9. Change the labels dictionary to match your model's labels
10. For each model, change the model_name variable to the name of the model folder (Tensorflow, tfLite, or TensorRT) and run all the blocks. This will:
    1. Create text files for predictions (needed for evaluation software)
    2. convert the result to mp4 if it's saved as avi (not needed but still there)
11. Open the evaluation software provided by Rafael Padilla
12. For Ground Truths:
    1. Set the Annotations option to the annotations folder in evaluation_stuff
    2. Set the Images option to the images folder in evaluation_stuff
    3. Click PASCAL VOC (.xml) for the Coordinates format
13. Click the "show ground-truths statistics" button and scroll through the images to make sure the ground truth boxes allign with the image
17. For Detections:
    1. Set the Annotations option to the predictions folder under the desired model folder within evaluation_stuff
    2. Leave the classes option empty
    3. Select the left middle option for Cordinates format ("<class name> <confidence> <left> <top> <right> <bottom> (ABSOLUTE)")
    4. Set the Output option to the eavl_results folder under the desired model folder within evaluation_stuff
18. Click the "show detections statistics" button and scroll through some images to see if most of the predictions line up with the ground truth box
19. For Metrics, select everything unless you don't want specific things. Our testing was done at an IoU threshold of 0.45.
20. After clicking run:
    1. The program will display a paragraph of metrics. Copy the text and put in a file named metrics.txt in the model folder within evaluation_stuff
    2. The program will fill the eval_results folder with pictures of each class's Precision vs Recall Curve, along with a graph of all of the classes together.
21. This concludes the evaluation step. Now, if you kept the names of the models straight, you should have accurate and organized results.
## References
 - Nicholas Renotte's in depth [tutorial](https://www.youtube.com/watch?v=yqkISICHH-U)
 - Rafael Padilla's evaluation [software](https://github.com/rafaelpadilla/review_object_detection_metrics)
