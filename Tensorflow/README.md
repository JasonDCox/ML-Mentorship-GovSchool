## Tensorflow Implementation
### Workflow
In this implementation, we create two different python scripts. 
One is for desktop picture annotation and model training, while the other is for running the model on the NVIDIA Jetson.

## Desktop
### Required Module Installs
After creating a new python environment, run the install script to install and upgrade
 - opencv-python
 - pyqt5
 - lxml
 - wget
 - tensorflow
 - protobuf
 - matplotlib
 - Pillow
 - pyyaml
 - pytz
#### CUDNN install
Determine the corresponding CUDNN version with your tensorflow version (latest) using this [website](https://www.tensorflow.org/install/source_windows).
Next, go through the CUDNN install process, which can be found in this [video](https://youtu.be/yqkISICHH-U?t=7001).
#### Verification Script
After installing all of this, you should be able to run the included verification script. If this runs successfully, the environment is ready for training. 
Otherwise, check the our error documentation or look up the error (they typically require uninstalling and reinstalling a module).
### Setting up for training
#### Personal pet
Run the gather images script to start taking images of your dog through the webcam, then go through the images folder and delete and blurry or useless ones. 
Next, run the annotation script, which should open the annotation tool. 
Now, go through each image and draw a box around your dog and save each image as you go.

#### General pet detection
If you would like to train for general pet detection rather than using the included model, download the image dataset as a tfrecord and place it into the annotations folder in the Tensorflow workspace.
Remane it test.record or train.record.

#### Creating the label map and creating the TFrecord
To create the new label map, go into the create label map script and labels list to match the classes of the annotated pictures or downloaded annotated data. 
If you did create your own annotated pictures, you will have to run the create TFrecord script. 
Lastly, run the create config file to finish the preperation steps for training

### Training
Run the training script, to which you can input how many training steps you want.

#### Post-training evaluation
After training, you can either run the evaluation script script to print out the model results, or you can run the tensorboard script to open up the interactive tensorboard interface.

### Detection
Run the desktop detection script, which will ask if you want to detect from images on your computer or from a webcam. It will then show the resulting image from the image detection or webcam stream.

### Exporting as TF_lite for the NVIDIA Jetson
Run the export script to export the model as a TensorflowLite model. This will create a folder of the tf_Lite model within the tensorflow model folder. 
You must copy this folder onto a flashdrive to transfer it to the NVIDIA Jetson.

## NVIDIA Jetson
### Importing

### Running

### Evaluating
