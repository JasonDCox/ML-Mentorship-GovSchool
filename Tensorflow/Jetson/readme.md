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
## Steps
### Exporting from Desktop
### TF-TRT
### tfLite
### Exporting back to Desktop
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
