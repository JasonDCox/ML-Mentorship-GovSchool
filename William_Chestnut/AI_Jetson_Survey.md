## 1
This paper has the objective of bringing together many different sources and summarizing them to give a large overview of the research done on the Nvidia Jetson platform. The paper starts by giving a brief background on embedded systems for machine learning with the general benefits of embedded systems, how they are optimized, and the specifications for different systems (TK1, TX1, TX2, Raspberry Pi 3(B+), and Intel UP). It then goes into a detailed overview of how people optimize their algorithms for these platforms by frame filtering, lowering resolution, reducing amount of depth-map levels, reducing neural network size with knowledge distillation, using smaller neural networks, and choosing an optimal network for the task. In the last section, this paper goes over many applications of embedded system machine learning in areas including medical, agricultural, robotics, autonomous driving, and drone navigation. 

## 2
One of the conclusions Mittal reaches is that depending on the programming language and package, the Nvidia Jetson platform can suffer from “model-load latency.” It was noted that this latency was especially bad with PyTorch, but it varies along with memory and energy consumption depending on the package.

The paper also suggests how the Jetson platform can be improved on a circuit level and microarchitecture level. Mittal suggests using smaller feature sizes to increase integration density and non-volatile memories to achieve near-zero idle power. He also suggests closer integration between the CPU and GPU at a microarchitecture level to reduce data-transfer overheads.

## 3
H. Halawa, H. A. Abdelhafez, A. Boktor, M. Ripeanu, “NVIDIA Jetson Platform Characterization,” The University of British ColumbiaVancouverCanada, 2017.

Supports the notion that higher-level programming languages such as Python may affect performance.
