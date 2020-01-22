 # Neural architecture search with network morphism
 This is prototype version, so the code is very messy and needs reorganization.
 
 ## Description
My old project on neural architecture search for the task of skin lesion analysis. The code is very messy and needs reorganization. I adopted the hill-climbing search strategy along with network morphism operations to explore the search space. I replicate the system described in “Simple And Efficient Architecture Search for Convolutional Neural Networks” by Thomas Elsken, Jan-Hendrik Metzen, and Frank Hutter (https://arxiv.org/abs/1711.04528), and modified some elements. The network morphism operations allow for incremental increases in the network size with the use of the previously trained network. The system was tested on the task of skin lesion diagnosis, which is a two-class classification problem (benign vs malignant). I used data from ISIC challenge.

 ## Dependencies
 - Tensorflow-gpu 1.12.0
 - Numpy
 - Graphviz
 - Pydot
 - GPUtil
## How to use
 Split your data into train, validation and test folders. In each folder, two folders with benign and malignant lesions should be placed. Provide the path to folders in main.py script.

 The results were presented in a conference paper:
 https://ieeexplore.ieee.org/abstract/document/8864624
 However, the code is to an extended version of the paper that is under review at the moment.  
If you find my work helpful please cite my paper:
```
@inproceedings{kwasigroch_deep_2019,
	title = {Deep neural network architecture search using network morphism},
	doi = {10.1109/MMAR.2019.8864624},
	booktitle = {2019 24th {International} {Conference} on {Methods} and {Models} in {Automation} and {Robotics} ({MMAR})},
	author = {Kwasigroch, Arkadiusz and Grochowski, Michal and Mikolajczyk, Mateusz},
	year = {2019},
	keywords = {deep neural networks, machine learning, image
    processing, neural architecture search, neural structure
    optimization},
	pages = {30--35}
}
```
