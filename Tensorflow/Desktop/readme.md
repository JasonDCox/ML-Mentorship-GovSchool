# Desktop Operation
## Layout
## Steps
### Creating, training, and exporting a model
1. Create/import image dataset with PASCAL VOC annotations.
  - creating/annotating custom images is covered in Nicholas Renotte's tutorial
  - there are custom code segments to help ensure imported annotations are in the correct order/format (it is very specific)
2. Create label map, tfrecords, and update the config file
  - change the label's array to match the exact names of your model's classes
  - there are code segments for all of these and Nicholas Renotte's video covers them in depth
3. Run Training Script
  - it is recomended to have 2000 training steps per class
  - it is also recomended to copy the command into terminal to see the progress as it goes
4. Run Evaluation Script and tensorboard
  - this will ensure the model is up to the performance you want and has fully flattened out (developed)
  - both of these are covered in Nicholas Renotte's video (tensorboard command not in notebook)
5. Run the detect on images and video
  - First run the segments under Load Train Model From Checkpoint
    - you must update checkpoint number to be the latest in the folder
  - Next run the segments under the Detect from all Test Images to run through all the images and log the detections for evaluation
  - Lastly, run the segments under Making a prediction video if you want to insert detections into a pre-filmed video
6. Export the model
  - run the code segment under Load Train Model From Checkpoint¶
### Evaluating results for models
## References
 - Nicholas Renotte's in depth [tutorial](https://www.youtube.com/watch?v=yqkISICHH-U)
 - Rafael Padilla's evaluation [software](https://github.com/rafaelpadilla/review_object_detection_metrics)
