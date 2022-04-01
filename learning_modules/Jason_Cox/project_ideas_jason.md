## Idea 1: Recognition/Security Doorbell

This doorbell would be able to recognize if someone walking in front of a house was a regular visitor, registered person, or some kind of animal and if so it would ignore it (unless the user wanted to warn about animals as well). However, if it detects an unusual face it could give an automatic update to the user with video warning of a possible intruder. 

Could also possibly detect if a package is being carried to the door and alert the user of this information. Or hypothetically if somone is walking away with a package the system could also warn of a robbery.Additionally, this could be used to allow a regular visitor to enter the house without a key  A project like this could be updated with new ideas as we have them and continue to evolve. 

### Hardware:
- Camera 

On the downside, I did see a few projects that were similar to this one, such as [this example](https://medium.com/@ageitgey/build-a-face-recognition-system-for-60-with-the-new-nvidia-jetson-nano-2gb-and-python-46edbddd7264), and it may not be as good because of that.     

Literature on facial recognition implementation - https://www.researchgate.net/profile/M-Meenakshi-2/publication/271338975_Real-Time_Facial_Recognition_System-Design_Implementation_and_Validation/links/5807a0d008ae63c48fec71bb/Real-Time-Facial-Recognition-System-Design-Implementation-and-Validation.pdf

## Idea 2: Fall Detection for Elderly

This idea would consist of a camera that could be  installed on something like a walking path for a retirement home and could detect if a person had fallen, possibly by registering their position (standing or laying down), and maybe how quickly they went down. This could then be connected to some sort of automated alarm system

### Hardware: 
- Camera

Literature on camera-based fall detectio: https://www.mdpi.com/1424-8220/17/12/2864/htm

I saw a few works in the NVIDIA Community Projects that serve as an example of registering the pose of a human through a camera, like [this one](https://neuralet.com/article/pose-estimation-on-nvidia-jetson-platforms-using-openpifpaf/), that could be useful in this project


## Idea 3: Disease prediction
This idea involves taking data and training the computer to predict if any coorelation exists between symptoms and the stored data. This could then serve to give faster warning for something like heart disease. Additionally this could be implemented within a device that already is collecting the data needed such as heartrate and bloodpressure to give automated warnings if something is not right. 

### Hardware: 
- None unless wanting to implement into existing systems

Literature on heart disease prediction: https://ieeexplore.ieee.org/abstract/document/8740989
