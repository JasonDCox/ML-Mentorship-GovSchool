# Tensorflow Implementation
## Overview
This folder contains the codes created for the project. **It does not include the desktop code from Nicholas Renotte's tutorial** on Tensorflow object detection on the desktop. It also does not include the program made by Rafael Padilla to evaluate the Average Precision of each model.

Each folder contains a readme file on installing and running everything.

## General Workflow
1. Create and Traing Model on Desktop
2. Run Tensorflow detections on Desktop
3. Export models for use on the Jetson (tfLite and normal export)
4. Transfer the models to the Jetson
5. Run the detections on the Jetson
6. Transfer the detection results back to the Desktop
7. Run Average Precision evaluation program and copy the metrics results text
