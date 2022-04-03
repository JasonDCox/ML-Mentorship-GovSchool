# YOLO v4 Implementation

## Description 
This implementation will be seperated between training and testing/usage sections. The training will be done with an Amazon Web Services ec2 instance and the testing on the 2GB NVIDIA Jetson Nano. 

## Requirements 
- CMake >= 3.18: https://cmake.org/download/
- Powershell: https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell
- CUDA >= 10.2: : https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
- OpenCV >= 2.4: download from [OpenCV official site](https://opencv.org/releases/) (on Windows set system variable OpenCV_DIR = C:\opencv\build - where are the include and x64 folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
- cuDNN >= 8.0.2: https://developer.nvidia.com/rdp/cudnn-archive (on Linux follow steps described here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on Windows follow steps described here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows)
- GPU with CC >= 3.0: https://en.wikipedia.org/wiki/CUDA#GPUs_supported

*note: These dependencies should already exist on the Jetson and correct AWS device. The previous steps are only required if they are missing or if you are attempting to use another system. 

## Training 

### Necessary Applications:
- [FileZilla](https://filezilla-project.org/download.php?show_all=1) or other file transfering application 
- [Putty](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) or other application to access the AWS device
- labelimg or other image labeling software

### Labelling the images
1. Install labelimg software in command line with `pip install labelimg`
2. Collect all pictures of the unique dog in one folder. For my dataset, this folder contained about 100 photos. 
3. Open folder in labelimg with command `labelimg [path to folder]` 
4. Ensure that the format below the Save button is YOLO. Otherwise, click the format until YOLO appears.
5. Within the labelimg tab, select the Change Save Dir button and select a folder to contain the pictures and their labels. 
6. Now press W to create a new box and draw this box around the head of the pet. Name the new label, click Save, and then Next Image. Repeat this process for the entire folder. 
7. Finally, because the software detects your unique label as the only one, it will call that label '0' in the new txt files. However, this will actually be label '37' once we add in the other data. This is soved using a python script similar to the one below. 

```
import glob, os

for name in glob.glob(os.path.join("C:\\Users\\jason\\Documents\\obj\\training", "20*txt")):

  
    nameOut = name.replace("training", "fixed")
    print(name + " " + nameOut)
    file = open(name, "r")
    output = open(nameOut, "w")

    x = file.readline()
    y = ("37" + x[1:])
    output.write(y)

    print(nameOut)
    file.close()
    output.close() 
```
The pathname "C:\\Users\\jason\\Documents\\obj\\training" and "20*txt" will need to be changed to whatever folder your labelled images are stored in and whatever similarities the filenames start with. Make sure to create a folder named fixed wherever you are running this program. This code will fill this "fixed" folder with new txt files correctly labelled. Replace the original files with these. 


### Establishing AWS ec2 Instance 
- placeholder 

### Training
1. Once connected to the AWS system, clone the darknet repository with `git clone https://github.com/AlexeyAB/darknet` 
2. For organization, create a 'yolov4-tiny' folder and a 'training' folder inside of that.
3. From [this link](https://public.roboflow.com/object-detection/oxford-pets/2) download the by-breed dataset in YOLO Darknet TXT format.  
4. Extract the 'train' folder from the download and upload it, using Filezilla, into the data folder of the darknet directory and rename to 'obj'
5. Within the darknet/cfg folder, edit the yolov4-tiny-custom.cfg file. 
    - change line batch to batch = 64
    - Change line subdivisions to subdivisions = 16
    - Set network size width = 640 height = 640, or any value multiple of 32
    - change line max_batches to (classes*2000 but not less than the number of training images and not less than 6000) This would be 38 * 2000 or 76000 for our dataset. 
    - change line steps to 80% and 90% of max_batches  
        
    - ![cfg_1](https://miro.medium.com/max/1400/1*t6MdI6CmJ7IGAh4v8_sRtw.png)
     - change [filters = 255] to filers = (classes + 5) * 3 (or 129 for us) in the 2 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.
     - Change line classes = 80 to the number of classes we have (38) in each of the 2 [yolo] layers
     - ![cfg_2](https://miro.medium.com/max/1400/1*hpqEFJDW9v0-hXGOmHOHeA.png)

6. Create "obj.data" and "obj.names" text files and upload them to the data folder within darknet
- obj.data:
```
classes = 38
train  = data/train.txt
valid  = data/test.txt
names = data/obj.names
backup = /yolov4-tiny/training
```
- obj.names comes from the label file in the downloaded dataset. Make sure to replace the last name with whatever was used to label your unique pet. 
```
cat-Abyssinian
cat-Bengal
cat-Birman
cat-Bombay
cat-British_Shorthair
cat-Egyptian_Mau
cat-Maine_Coon
cat-Persian
cat-Ragdoll
cat-Russian_Blue
cat-Siamese
cat-Sphynx
dog-american_bulldog
dog-american_pit_bull_terrier
dog-basset_hound
dog-beagle
dog-boxer
dog-chihuahua
dog-english_cocker_spaniel
dog-english_setter
dog-german_shorthaired
dog-great_pyrenees
dog-havanese
dog-japanese_chin
dog-keeshond
dog-leonberger
dog-miniature_pinscher
dog-newfoundland
dog-pomeranian
dog-pug
dog-saint_bernard
dog-samoyed
dog-scottish_terrier
dog-shiba_inu
dog-staffordshire_bull_terrier
dog-wheaten_terrier
dog-yorkshire_terrier
unique_pet
```

7. Upload this process.py script file to the darknet directory and run it to create files for training and testing data. This script will note paths to all images in the obj folder for the algorithm to use. 

```
import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

current_dir = 'data/obj'

# Create and/or truncate train.txt and test.txt
file_test = open('data/test.txt', 'w')
file_train = open('data/train.txt', 'w')


for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    file_train.write("data/obj" + "/" + title + '.jpg' + "\n")

```

8. Make changes in the Makefile 
    - Change GPU = 0 to GPU = 1 
    - Change CUDNN = 0 to CUDNN = 1
    - Change CUDNN_HALF = 0 to CUDNN_Half = 1
    - Change LIBSO = 0 to LIBSO = 1
    
9. Run `make` command to build darknet.

10. Download the pre-trained YOLOv4-tiny weights with command `wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29` on AWS system to use as a starting point. 

11. Run the trainig command 

```
nohup ./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map >& outlog.txt &
```

You can use the command, `tail -f outlog.txt`, to monitor the progress. 

If the training did not complete, you can restart the training from the last saved checkpoint with the command: 

```
nohup ./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg /yolov4-tiny/training/yolov4-tiny-custom_last.weights -dont_show -map >& outlog.txt &
```

You should now be able to find and download from the AWS system all the trained weights from the darknet/yolov4-tiny/training folder. 

Additionally download the yolov4-tiny-custom.cfg file you edited, the obj.data file and the obj.names file. 

### Testing and Usage
1. On Nvidia Jetson Nano, clone the darknet repository with `git clone https://github.com/AlexeyAB/darknet' 
2. Transfer the training folder with trained weights into the darknet directory 
3. Within the darknet/cfg folder, replace the yolov4-tiny-custom.cfg file with the one you edited earlier. 
4. Add the obj.data and obj.names files that you donwloaded to the darknet/data folder. 
5. Upload a folder named "obj" with all of the photos you wish to test with. This can be comprised of the "test" folder from the previously installed zip file as well as approximately 20 additional, properly labelled and combined,photos of the unique pet. Remember to correctly update the labeled index as we did earlier with this script before uploading: 

```
import glob, os

for name in glob.glob(os.path.join("C:\\Users\\jason\\Documents\\obj\\testing", "20*txt")): #Change to correct path and remove old "fixed" files

  
    nameOut = name.replace("testing", "fixed")
    print(name + " " + nameOut)
    file = open(name, "r")
    output = open(nameOut, "w")

    x = file.readline()
    y = ("37" + x[1:])
    output.write(y)

    print(nameOut)
    file.close()
    output.close() 
```
6. Upload this slightly updated process.py script file to the darknet directory and run it. 

```
import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

current_dir = 'data/obj'

# Create and/or truncate train.txt and test.txt
file_test = open('data/test.txt', 'w')
file_train = open('data/train.txt', 'w')


for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    file_test.write("data/obj" + "/" + title + '.jpg' + "\n")

```

7. Make the following changes to the Makefile:
    - Change GPU = 0 to GPU = 1 
    - Change CUDNN = 0 to CUDNN = 1
    - Change CUDNN_HALF = 0 to CUDNN_Half = 1
    - Change OPENCV = 0 to OPENCV = 1
    - Change LIBSO = 0 to LIBSO = 1\
    - Replace ARCH = section with ARCH = -gencode arch=compute_53, code=sm_53 

8. Run `make` command to build darknet.

9. Run ``` ./darknet detector map data/obj.data cfg/yolov4-tiny-custom.cfg training/yolov4-tiny-custom_*****.weights ``` for every iteration of weights and look for the "mean average precison" to determine which was most effective. Note these weights for future use. 

You have now successfully trained, tested, and identified the best weights to detect a unique pet. Refer to [this](https://github.com/AlexeyABdarknet#how-to-use-on-the-command-line) github repository and specifically the "how to use command line" section of the readme for details on how to use these weights. 
