# Jetson Operation
## Overview
The Jetson Operation will run both tfLite and TensorRT. tfLite has a simple installation process and TensorRT should come preinstalled with the Jetson Nano, but I ran into an issue where it had to be reinstalled.

Once the models have been fully trained and exported on the Desktop, you will copy them into the specified locations in the object_detection folder. From there, you can transfer the object_detection folder onto the Jetson and run the different evaluation scripts. This will go through all the images, make predictions, and log these predictions for later analysis on the desktop. The scripts also have the option of making predictions on a video and making predictions through a camera feed. Once you are done running everything on the Jetson, transfer the results back to the Desktop to analyze the Average Precision of the two models.

It is important to note that **the TensorRT script uses TF-TRT,** which is a method of leveraging TensorRT that has an extremely high load-in price. On a device as limited as the Jetson, this results in 20+ minute load times for the model as TF-TRT may roll over into SWAP memory as it loads the entirety of Tensorflow. There is a better way of utilizing TensorRT that doesn't require the high memory overhead, which is to create a TensorRT engine, but it requires a different approach that is not covered in this github.

## Installation
1. I recommend installing the Jetson Stats application to help turn on the fan, monitor multiple system usage stats, and look at the versions of the OS, TensorRT, CUDA, and CUDNN. This installation instructions are in their github, which is linked below.
2. The Installation for tfLite is simple and covered in Nicholas Renotte's video. His RasberryPi code github is linked below and contains a list of the required modules.
3. To install tensorflow, follow the instructions from NVIDIA, which are linked below.
4. If you keep having issues with TensorRT wanting to use a "libnvinfer" that does not exist, try reinstalling TensorRT using NVIDIA's instructions that are linked below.
## Layout
1. The final resulting layout for the folders used in this section of the tutorial should be the same as in the object_detection folder from the github
## Steps
### Exporting from Desktop
1. After running the graph freezing and tfLite export section of the desktop code, do the following steps to transfer the model
2. Go into TFODCourse > Tensorflow > Workspace > models > "you model name" > export
3. Copy all of the files and folders (checkpoint, saved_model, pipline.config)
4. Paste it into the folder: object_detection > TF-TRT > model
    1. This is it folder TensorRT will draw the model from
5. For tfLite: Go into TFODCourse > Tensorflow > Workspace > models > "you model name" > tfliteexport > saved_model
6. Copy the detect.tflite file
7. Paste it in the folder: object_detection > tfLite
8. Rename the version in object_detection to model.tflite
9. Now copy the object_detection folder to the Jetson for use
### TF-TRT
1. Now that everything is on the Jetson, open the command terminal
2. Before you run the tensorRT_test.py script, you must edit/change the input resolution (currently 640, 640) to the input resolution of your model
    1. replace all occurences of 640, 640
3. To run the program, cd into the folder: object_detection > TF-TRT
5. Run the tensorRT_test.py script
    1. command: python tensorRT_test.py
6. The program will ask if you would like to opimize the model, only say yes if you have not optimized the model yet
    1. Once the model has been optimized, tensorRT saves the optimized version, so this step can be skipped
7. Select your runtime method (c for camera, i for image folder, and v for video)
8. If you choose camera:
    a. The program will load the model and open a feed of the first camera
    b. It will then start displaying predictions on the feed window at the frame rate of the model
    c. Press q to exit the feed window
9. If you choose image folder:
    1. The program will load the model
    2. It will then go through all the images in the test_images folder. For each image it will:
        1. It will make predictions for the image
        2. It will draw the highest confidence value prediction onto the image and save it into the result_images folder
            a. It will print the class of the highest confidence object
        3. It will add the predictions to a dataframe of image predictions
    3. It will save the image predictions dataframe to the tenorRT_results.csv
    4. It will print the average FPS excluding the first 5 images where the model is warming up
10. If you choose video:
    1. The program will load the model
    2. It will then open the video "test.mp4"
    3. It will go through each frame, draw all predictions above a 60% confidence interval, and save it to a video called "tensorRT.mp4"
    4. The resulting video will be 10fps with each frame having a prediction run on it, so it does not smulate the speed of the model
### tfLite
1. Now that everything is on the Jetson, open the command terminal
2. Before you run the evaluate.py script, you must edit/change the input resolution (currently 640, 640) to the input resolution of your model
    1. replace all occurences of 640, 640
3. cd into the folder: object_deteciton > tfLite
4. Run the evaluate.py script
    1. command: python evaluate.py
5. Select your test option: i for test images, v for video, or c for camera feed
6. If you select test images:
    1. The program will load the tfLite model
    2. It will then go through all the images in the test_images folder. For each image it will:
        1. It will make predictions for the image
        2. It will draw the highest confidence value prediction onto the image and save it into the tfLite_results folder
            a. It will print the class of the highest confidence object
        3. It will add the predictions to a dataframe of image predictions
        4. It will print the current FPS for the image predictions
    3. It will save the image predictions dataframe to the tfLite_results.csv file
    4. It will print the average fps of the model
7. If you select video:
    1. The program will load the model
    2. It will then open the "test.mp4" video
    3. It will go through each frame, make and draw predictions above a 60% confidence threshold, and save them to the video "tfLite.mp4"
        a. It will list any classes it detects as it goes
    5. The resulting video will have predictions run on each frame and will be 10 fps, so it doesn't reflect the speed of the model.
8. If you select camera
    1. The program will load the model and open the first to the first camera
    2. It will then open a window with a feed of each frame with predictions at the speed of the model
    3. press q to close the feed window
### Exporting back to Desktop
1. Now that all your tests are completed, copy the object detection folder back to the Desktop pc
2. Follow the instructions in the desktop readme on how to evaluate and export the results from the models
## References
 - Nicholas Renotte's in-depth [tutorial](https://www.youtube.com/watch?v=yqkISICHH-U)
    - [RasberryPi code github](https://github.com/nicknochnack/TFODRPi)
    - Time stamps:
    - a
    - a
    - a
    - a
    - a
    - a
    - a
    - a
    - a
    - a
    - a
    - a
 - NVIDIA [Tensorflow install instructions for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
 - NVIDIA [TensorRT install instructions for Jetson](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
 - NVIDIA [tegrastats documentation](https://docs.nvidia.com/drive/drive_os_5.1.6.1L/nvvib_docs/index.html#page/DRIVE_OS_Linux_SDK_Development_Guide/Utilities/util_tegrastats.html)
     - This provides the best way of accurately monitoring the memory, CPU, and GPU usage of the Jetson under heavy load.
 - [jetson stats github](https://github.com/rbonghi/jetson_stats)
     - This provides a great way of turning on the fan for the Jetson, monitoring multiple system usage stats, and looking at the versions of the OS, TensorRT, CUDA, and CUDNN. However, the program crashes when the Jetson is under heavy load, which is why I suggest using tegrastats to monitor system usage during testing.
