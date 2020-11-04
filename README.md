# LearningSequences
Learning Sequences of Dermoscopic Attributes for Explainable Skin Lesion Diagnosis code. Final master project. 

The main code is written in jupyter notebooks (.ipynb), while all the auxiliar functions are written in python scripts (.py).

For being able to run the code properly, firstly, the path has to be changed to the one you are going to use. For this purpose, the variable "global_path" within the "m3.ipynb" has to be initialized with the rute to the file of your preference. The files "m3.ipynb", "common_code" and "data_2017" have to be inside. Then, all the additional files and outputs will be created inside this folder too.

The ISIC 2017 database has to be organized in the "data_2017" file as follows:
- Three folders: "Train", "Val" and "Test" with the dermoscopic and superpixels images of each of the sets.
- Three folders: "Train_Dermo", "Val_Dermo" and "Test_Dermo" with the .json files which contain the dermoscopic structures files for each of the sets.
- Three folders: "Train_Lesion", "Val_Lesion" and "Test_Lesion" with the binary masks of the lesions for each of the sets. 
