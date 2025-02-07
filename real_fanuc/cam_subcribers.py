import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CameraSubscriber(Node):
    def __init__(self, serials):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()
        self.subscribers = []
        self.windows = {}

        for serial in serials:
            if serial =="332322070892":
                cam_name='gripper_top'
            elif serial =="332522076772":
                cam_name='top'
            topic_name = f'/camera_{cam_name}/image_raw'
            self.get_logger().info(f'Subscribing to {topic_name}')
            
            sub = self.create_subscription(
                Image,
                topic_name,
                lambda msg, serial=serial: self.image_callback(msg, serial),
                10
            )
            self.subscribers.append(sub)

            window_name = cam_name
            self.windows[serial] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def image_callback(self, msg, serial):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imshow(self.windows[serial], cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error converting image from {serial}: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    # 替换成你相机的实际序列号
    serials = ['332322070892',"332522076772"]  # 示例相机序列号
    node = CameraSubscriber(serials)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down camera subscriber...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
