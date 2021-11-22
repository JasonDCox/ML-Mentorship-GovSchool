## Animals on Road Detection
This would be a system which detects and indetifies animals on the blind turn at the beginning of Butler Farm Road. If it detects animals on the road, a sign 100 feet before the turn will start blinking.

### Parts
 - Light Camera
 - Infarared Camera
 - Solar Power system
### Software
 - Convolutional Nueral Network with an algorithm like YOLO since detail is not very important in this scenario
### Literature
- A Proposal of an Animal Detection System Using Machine Learning
    - https://www.tandfonline.com/doi/abs/10.1080/08839514.2019.1673993

![picture of turn](Map_pic.png)


## Advanced Radio
Radio that detects if a song matches a set of unwanted songs and changes the station if that is the case. We could also go further and recomend stations that are playing liked songs or changes stations when it detects an ad.

### Parts
 - AUX input
 - Radio that is willing to be modified heavily (for simulated buttons)
### Software
 - Create a Spectrogram of the audio and compare it with others in a database (most likely similar to k-Nearest Neighbors)
### Literature
- Fog Computing Approach for Music Cognition System Based on Machine Learning Algorithm
    - https://www.dxomark.com/how-do-algorithms-listen-to-music/
- Houston Toad and Other Chorusing Amphibian Species Call Detection Using Deep Learning Architectures
    - https://ieeexplore.ieee.org/abstract/document/9031223


## CNC Safety System
A common problem with the CNC at my FRC robotics team is that polycarbonate may melt onto the bit and pieces may stick onto the bit when it finishes cutting them completely. Similar to the 3d print failure detection system, this project would utilize a camera mounted to directly view the CNC bit to alert the machine shop and turn off the CNC bit when a failure/danger occurs.
### Parts
 - Camera
 - Custom mount for CNC
### Software
 - Convolutional Nueral Network with an Object detection algorithm that is decent with detail since contrast may be low in this scenario
### Literature
- Real-time defect detection in 3D printing using machine learning
    - https://www.sciencedirect.com/science/article/pii/S2214785320381037
- Detection of Material Extrusion In-Process Failures via Deep Learning
    - https://www.mdpi.com/2411-5134/5/3/25


