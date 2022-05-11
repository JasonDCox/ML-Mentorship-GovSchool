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
    3. Lastly, run the segments under Making a prediction video if you want to insert detections into a pre-filmed video
6. Export the model
    1. Run the code segment under Load Train Model From Checkpoint
### Evaluating results for models
## References
 - Nicholas Renotte's in depth [tutorial](https://www.youtube.com/watch?v=yqkISICHH-U)
 - Rafael Padilla's evaluation [software](https://github.com/rafaelpadilla/review_object_detection_metrics)
