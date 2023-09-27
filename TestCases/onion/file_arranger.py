import os
import sys
import shutil

class file_arranger:
    def __init__(self,file_path,file_name):
        try:
            self.file_path = file_path
            self.file_name = file_name
            
        except Exception as error:
            print("Exception occur due to {0}".format(str(error)))
            sys.exit(1)
    
    def arrange(self):
        try:
            for key in list(self.file_path.keys()):
                for file in list(self.file_name.keys()):
                    target_file = self.file_path[key]['files_location'] +self.file_name[file]
                    destination = self.file_path[key]['target_location']+file+"/"
                    image_location = self.file_path[key]['images_location']
                    annotation_location = self.file_path[key]['Annotation_location']
                    with open(target_file,'r') as file:
                        file_names = file.readlines()
                    file_names =[name.strip() for name in file_names]
                    if os.path.exists(destination):
                        shutil.rmtree(destination)  
                    os.makedirs(destination)
                    for file in file_names:
                        image_path = image_location+file+".jpg" 
                        annotation_path = annotation_location+file+".txt"
                        shutil.copy2(image_path,destination)
                        shutil.copy2(annotation_path,destination)
        except Exception as error:
            print("Exception occur due to {0}".format(str(error)))
            sys.exit(1)

if __name__ =="__main__":
    
    file_path_info = {"balanced":{"files_location":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_balanced/onion_balanced/ImageSets/Main/",
                                  "images_location":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_balanced/onion_balanced/JPEGImages/",
                                  "target_location":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_balanced/",
                                  "Annotation_location":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_balanced/modified_Annotation/"},
                      "unbalanced":{"files_location":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_unbalanced/onion/ImageSets/Main/",
                                    "images_location":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_unbalanced/onion/JPEGImages/",
                                    "target_location":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_unbalanced/",
                                    "Annotation_location":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_unbalanced/modified_Annotation/"}
                      }
    
    file_name_info ={"training":"train.txt",
                    "testing":"test.txt",
                    "train_validation":"trainval.txt",
                    "test_validation":"val.txt"
                    }
    file_praser = file_arranger(file_path_info,file_name_info)
    file_praser.arrange()
                    