import os
import sys
import xml.etree.ElementTree as ET

class praser:
    def __init__(self,path,laning_location):
        try:
            self.path=path
            self.proccessed_landing_location = laning_location
            self.files = os.listdir(self.path)
            self.__make_dir()
        except Exception as error:
            print("Exception occur due to {0}".format(str(error)))
            sys.exit(1)
            
    def __make_dir(self):
        try:
            if not os.path.exists(self.proccessed_landing_location):
                os.makedirs(self.proccessed_landing_location)
            print("we have all target directory available")
        except Exception as error:
            print("Exception occur due to {0}".format(str(error)))
            sys.exit(1)
            
    def traversing(self):
        try:
            classification_label = {"with_weeds":0,"without_weeds":1}
            for file in self.files:
                file_location = self.path+"/"+file
                tree = ET.parse(file_location)
                root = tree.getroot()
                # fetching the image width information
                image_size_width_element =  root.findall(".//size/width")
                image_width = int(image_size_width_element[0].text)
                
                image_size_height_element =  root.findall(".//size/height")
                image_height= int(image_size_height_element[0].text)

                with open(self.proccessed_landing_location+file.split(".")[0]+".txt", "w") as final_txt_file:
                    for object_information in root.findall("object"):
                         classification_name = object_information.find('name').text
                         classification_name=classification_name.strip().lower()
                         classification_name = "_".join(classification_name.split(" "))
                         if classification_name not in list(classification_label.keys()):
                             raise Exception("Label information not found")
                         label_value = classification_label[classification_name]
                         for bounding_box in object_information.findall('bndbox'):
                             xmin = int(bounding_box.find("xmin").text)
                             xmax = int(bounding_box.find("xmax").text)
                             ymin = int(bounding_box.find("ymin").text)
                             ymax = int(bounding_box.find("ymax").text)
                             width = xmax-xmin
                             height = ymax-ymin
                             x_center = (xmax+xmin)/2
                             y_center = (ymax+ymin)/2
                             x_center_normalized = x_center/image_width
                             width_normalized = width/image_width
                             y_center_normalized = y_center/image_height
                             height_normalized = height/image_height
                             value = str(label_value)+" "+str(x_center_normalized)+" "+str(y_center_normalized)+" "+str(width_normalized)+" "+str(height_normalized)
                             final_txt_file.write(value+ '\n')
        except Exception as error:
            print("Exception occur due to {0}".format(str(error)))
            sys.exit(1)
                 
if __name__ == "__main__":
    path_info = {"balanced":
                             {"source_path_info":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_balanced/onion_balanced/Annotations",
                              "destination_path_info":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_balanced/modified_Annotation/"
                                },
                "unbalanced":
                            {"source_path_info":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_unbalanced/onion/Annotations",
                             "destination_path_info":"C:/Paolo-Project-Data-Models/Onion Datasets/onion_unbalanced/modified_Annotation/"
                               },
                }
    for key in list(path_info.keys()):
        xml_praser = praser(path_info[key]['source_path_info'],path_info[key]['destination_path_info'])
        xml_praser.traversing()
        print("Modification process completed for {0}".format(str(key)))