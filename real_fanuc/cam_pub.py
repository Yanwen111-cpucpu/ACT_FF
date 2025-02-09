# realsense_camera.py (ROS 1 Version)
import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class RealSenseCamera:
    """ 每个 RealSense 相机的 ROS 1 发布节点 """
    
    def __init__(self, serial):
        self.serial = serial
        self.bridge = CvBridge()
        
        if serial == "332322070892":
            cam_name = "gripper_top"
        elif serial == "332522076772":
            cam_name = "top"
        else:
            cam_name = f"camera_{serial}"

        # 话题名称：camera_{cam_name}/image_raw
        self.publisher = rospy.Publisher(f'camera_{cam_name}/image_raw', Image, queue_size=10)

        # 初始化 RealSense 相机
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 25)

        self.pipeline.start(config)
        rospy.loginfo(f'RealSense Camera {serial} started, publishing to camera_{cam_name}/image_raw')

    def capture_and_publish(self):
        """ 获取相机帧并发布到 ROS 1 话题 """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())

        # 转换 OpenCV 图像为 ROS 1 消息
        ros_image = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')

        # 发布到 ROS 1
        self.publisher.publish(ros_image)
        rospy.loginfo(f'Published image from camera {self.serial}')

    def stop_camera(self):
        """ 停止相机 """
        self.pipeline.stop()


def main():
    rospy.init_node('realsense_camera', anonymous=True)
    
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

    if not serials:
        rospy.logerr("❌ No Intel RealSense devices found!")
        return

    cameras = [RealSenseCamera(serial) for serial in serials]
    rate = rospy.Rate(25)  # 25Hz 采样频率
    
    try:
        while not rospy.is_shutdown():
            for cam in cameras:
                cam.capture_and_publish()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        for cam in cameras:
            cam.stop_camera()

if __name__ == '__main__':
    main()