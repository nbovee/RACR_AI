from data_wrapper import data_wrapper
from PIL import Image
import rospy
from std_msgs.msg import Image

## Inside ROS
# Cameras publishes to images/ ROS topic
# \/
## This code
# await roscore?
# connects to the image topic specified
# In the callback saves the latest frame in the "latest" variable as a PIL image.
# set "new image" flag true
# loop above replacing the previous frame as images are receieved
# call function basically needs to return new image, and sleep/loop until it can. sets "new_image" to false
# \/
## ML Code
# get image from ros_data (could incur delay)
# process
# loop

class ros_loader(data_wrapper):

    def callback(data):
        rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)

    def __init__(self):
        self.image_list = []
        self.new_image = False
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber('chatter', Image, self.callback)
    
    def has_next(self):
        return not rospy.is_shutdown()

    def next(self):
        if self.new_image is False:
            pass # sleep thread 1 ms here. does rospy have sleep?
        else:
            [val, filename] = self.image_list.pop()
            print(f"Testing \"{filename}\": # images left={len(self.image_list)}")            
            for i in range(1,21): # hardcoded split layers for AlexNet - no full processing yet. Layer 0 would be full server, 21 would be full client
                yield [ val, i, filename ]

    def load_data(self, path):
        max_images = 1000
        self.image_list.clear()
        # print(f"len iglob {len(list(glob.iglob(path)))}")
        for image in glob.iglob(path):
            # print(image)
            self.image_list.append([ Image.open(image).convert('RGB'), image ])
            if len(self.image_list) >= max_images:
                break