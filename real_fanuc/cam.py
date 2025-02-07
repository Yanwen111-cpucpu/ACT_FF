import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class RealSenseCamera(Node):
    """每个 RealSense 相机的 ROS 2 发布节点"""
    
    def __init__(self, serial):
        super().__init__(f'realsense_camera_{serial}')
        
        self.serial = serial
        self.bridge = CvBridge()

        # 话题名称：camera_{serial}/image_raw
        self.publisher = self.create_publisher(Image, f'camera_{serial}/image_raw', 10)

        # 初始化 RealSense 相机
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
        self.get_logger().info(f'RealSense Camera {serial} started, publishing to camera_{serial}/image_raw')

        # 50Hz 采样频率（20ms ）
        self.timer = self.create_timer(0.02, self.capture_and_publish)

    def capture_and_publish(self):
        """获取相机帧并发布到 ROS 2 话题"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())

        # 转换 OpenCV 图像为 ROS 2 消息
        ros_image = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')

        # 发布到 ROS 2
        self.publisher.publish(ros_image)
        self.get_logger().info(f'Published image from camera {self.serial}')

    def stop_camera(self):
        """停止相机"""
        self.pipeline.stop()

def main(args=None):
    rclpy.init(args=args)

    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]

    if not serials:
        print("❌ No Intel RealSense devices found!")
        return

    # 创建多个相机 ROS 2 节点
    nodes = [RealSenseCamera(serial) for serial in serials]
    executor = rclpy.executors.MultiThreadedExecutor()

    for node in nodes:
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        for node in nodes:
            node.stop_camera()
            node.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()
