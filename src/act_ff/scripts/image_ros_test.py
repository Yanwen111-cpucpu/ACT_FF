# image_display.py (ROS 1 Version)
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from robot_utils import ImageRecorder

class ImageDisplay:
    """ 订阅 /camera_gripper_top/image_raw 和 /camera_top/image_raw 并显示图像 """
    
    def __init__(self):
        rospy.init_node('image_display', anonymous=True)
        self.bridge = CvBridge()
        self.image_recorder = ImageRecorder(init_node=False)
        
        rospy.loginfo('Subscribed to /camera_gripper_top/image_raw and /camera_top/image_raw')
        
    def show_images(self):
        rate = rospy.Rate(10)  # 10 Hz 刷新率
        while not rospy.is_shutdown():
            images = self.image_recorder.get_images()
            for cam_name, image in images.items():
                if image is not None:
                    cv2.imshow(cam_name, image)
            cv2.waitKey(1)
            rate.sleep()
        cv2.destroyAllWindows()


def main():
    display = ImageDisplay()
    display.show_images()

if __name__ == '__main__':
    main()
