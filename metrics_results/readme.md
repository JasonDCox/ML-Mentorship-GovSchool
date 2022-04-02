## Overview
This section has the results for each model.
This includes a text document of their metrics, a folder of their predictions on test data, and graphs of their Precision/Recall curves. Along with this, there are some more resource oriented metrics within the extra_metrics.md file for each model, except desktop tensorflow since it is not ran on the Jetson Nano

These extra metrics include:
 - Average FPS
 - Memory usage
 - CPU usage
 - GPU usage
 - Load time (if applicable/concerning)

The primary accuracy metrics revolve around the mAP for each class in the dataset at a IoU threshold of 0.45. The custom dog in this dataset is Bella, so the metric for her is of special interest in this case.
