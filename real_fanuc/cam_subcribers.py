import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CameraSubscriber:
    """ 订阅 RealSense 相机的 ROS 1 节点 """
    
    def __init__(self, serials):
        rospy.init_node('camera_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.subscribers = []
        self.windows = {}
        self.latest_images = {serial: None for serial in serials}

        for serial in serials:
            if serial == "332322070892":
                cam_name = "gripper_top"
            elif serial == "332522076772":
                cam_name = "top"
            else:
                cam_name = f"camera_{serial}"

            topic_name = f'/camera_{cam_name}/image_raw'
            rospy.loginfo(f'Subscribing to {topic_name}')
            
            sub = rospy.Subscriber(topic_name, Image, self.image_callback, callback_args=serial)
            self.subscribers.append(sub)

            window_name = cam_name
            self.windows[serial] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def image_callback(self, msg, serial):
        """ 处理接收到的 ROS 图像数据 """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_images[serial] = cv_image  # 存储最新图像
        except Exception as e:
            rospy.logerr(f"Error converting image from {serial}: {e}")

    def show_images(self):
        """ 循环显示订阅到的相机图像 """
        rate = rospy.Rate(20)  # 10 Hz
        while not rospy.is_shutdown():
            for serial, image in self.latest_images.items():
                if image is not None:
                    cv2.imshow(self.windows[serial], image)
            cv2.waitKey(1)  # 确保窗口刷新
            rate.sleep()
        cv2.destroyAllWindows()


def main():
    serials = ['332322070892', '332522076772']  # 替换为实际相机序列号
    sub = CameraSubscriber(serials)
    sub.show_images()

if __name__ == '__main__':
    main()
