## 1 Overview
The paper starts by explaining how visual motion tracking has become more reliable and accurate in applications where tracking motion with wheel rotation or other methods may not be as feasible and consistent. It then describes the basic visual odometry setups (monocular, stereo, and multi-camera) and how embedded systems for machine learning, such as the Nvidia Jetson, can be applied in this situation. Next, the paper explains the basics of visual motion tracking and the different algorithms involved with it, along with how they generally function. The researchers then test these algorithms, which include ORB-SLAM2, VISO2, RTAB-MAP, SPTAM, and ZED-VO, on  both CPU only mode and GPU assisted mode and record the resulting performance. These recordings include the Absolute Trajectory Error, CPU utilization, and FPS.

## 2 Conclusions
This paper concludes that ORB-SLAM2 and RTAB-MAP gave the best performance when considering accuracy and FPS. However, the paper does note that future improvements can be made to help increase the FPS for each algorithm.

The paper also claims that a possible option for increasing algorithm performance in terms of FPS would be to edit the source code of the algorithm to include “GPU implementation of feature extraction and description tasks.” However, they make it clear that the feasibility of this may be questionable.

## 3 Extension
S. Aldegheri, N. Bombieri, D. D. Bloisi, A. Farinelli, “Data Flow ORB-SLAM for Real-time Performance on Embedded GPU Boards,” Institute of Electrical and Electronics Engineers

Supports the notion that modifying the source code of algorithms such as ORB-SLAM2 can greatly improve their performance on the Nvidia Jetson platform.
