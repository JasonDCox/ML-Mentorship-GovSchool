# Implementation of Pet Detection and Recognition on the NVIDIA Jetson Nano
This repository houses code and documentation for a GSST mentorship project. We have taken up the task of creating a system that will automatically detect and recognize a pet for the purpose of controlling hardware systems. This repository will additionally be updated with experimental results and eventually include a final project paper.

## Problem Statements

### Societal:
A pet can approach the door to their home and be let in automatically. Currently, facial recognition is used for humans to access things such as their phones or open locks but this is not widely implemented in the same sense for animals and the most common solution to allowing free entry of pets is an unlockable door to the home. Most pets now have to rely on the attention and availablility of busy owners to let them out of possibly harsh conditions or face a security hazard with something like a conventional dog door. To avoid this, we plan to use the NVIDIA Jetson nano with software designed to automatically recognize a pet that matches what the user inputted into the program and unlock the door. This way, a user can add or remove pets and the door will only open to let in that specific pet. 

### Engineering
The proposed engineering problem is combining low power artificial intelligence/machine learning solutions with miniature, cost-effective resources. We will be investigating multiple approaches to the problem to gather results showing what would be most beneficial for the end user in future development of a physical product. 

## Minimum Viable Product
Process of experimental testing in which we are able to read in and train an algorithm on custom created datasets, and then detect a specific pet, with high accuracy, in future demonstrations. This process should be able to eventually be implemented in a control system. 


## Navigation 
- Media: Contains video used for testing.
- Tensorflow: Contains tutorials and code for the Tensorlow implementation.
- YOLO: Contains tutorials and code for the YOLO implementation.
- dataset: Contains all additional custom images used for testing. Does not include entire pet dataset.
- Learning_modules: Contains mentorship materials created before work on final project.
- Metrics_results: Contains final metrics on effectiveness of different implmentations.
- System Documentation: Contains all directories regarding documentation.
