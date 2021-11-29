# CLIENT REQUIREMENTS
### Description
The goal for this product is to enable pet owners to automate house access for their animals. Ideally, pets can enter and exit the building without owner participation. The key distinction is that animals that are not the owner's pets should be denied access to the building.  
### Functional 
- Mechanism for approving/denying access to the building. Likely implemented via AI/ML. 
- Mechanism for changing the list of approved animals that can unlock the door. 
- Mechanism for gathering pet data for the algorithm to learn on. 
- Notification system that provides details to the owners of the product. Such as:
	- Were any non-registered animals detected?
	- How many times was the entrance unlocked by each pet?
- Repeated apperiances of non-labeled entities should be cataloged. Optionally allow a user to apply a label. 
### Software
- Implemented on embedded computing system. Must be a low/no cost solution such as the Nvidia Jetson. 
### Hardware
- Devise a locking mechanism that relies on the output of your software to control the lock on the door. 
### User
- Must have documentation on how to use the application.
- Implement a user interface to carry out functional requirements above. 			