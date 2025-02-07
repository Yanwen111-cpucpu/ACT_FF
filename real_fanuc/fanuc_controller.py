import rclpy
from rclpy.node import Node
import socket
import struct
from std_msgs.msg import Float64MultiArray
import numpy as np

class FanucCmd(Node): # Sub cmd
    def __init__(self):
        super().__init__('fanuc_sender')
        self.declare_parameter('target_ip', '192.168.1.100')
        self.declare_parameter('target_port', 3827)
        self.declare_parameter('frequency', 50)
        
        self.target_ip = self.get_parameter('target_ip').value
        self.target_port = self.get_parameter('target_port').value
        self.frequency = self.get_parameter('frequency').value
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.subscription = self.create_subscription(
            Float64MultiArray, '/robot_cmd', self.send_udp_data, 10)
        self.get_logger().info(f'Listening for commands on /robot_cmd')
    
    def send_udp_data(self, msg):
        if len(msg.data) != 6:
            self.get_logger().warn('Received incorrect data length')
            return
        try:
            msg=msg/3.14*180
            packed_data = struct.pack('<6d', *msg.data)
            self.sock.sendto(packed_data, (self.target_ip, self.target_port))
            self.get_logger().info(f'Sent: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error sending data: {e}')


class FanucPub(Node): #Pub state
    def __init__(self):
        super().__init__('fanuc_receiver')
        self.declare_parameter('local_ip', '0.0.0.0')
        self.declare_parameter('port', 9600)
        self.declare_parameter('frequency', 50)
        
        self.local_ip = self.get_parameter('local_ip').value
        self.port = self.get_parameter('port').value
        self.frequency = self.get_parameter('frequency').value
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_ip, self.port))
        self.sock.settimeout(0.01)
        
        self.publisher = self.create_publisher(Float64MultiArray, '/robot_state', 10)
        self.timer = self.create_timer(1.0 / self.frequency, self.receive_udp_message)
        self.get_logger().info(f'Listening for UDP messages on {self.local_ip}:{self.port}')
    
    def receive_udp_message(self):
        try:
            data, _ = self.sock.recvfrom(36 * 8)
            if len(data) == 36 * 8:
                unpacked_data = struct.unpack('<36d', data)
                data_array = np.array(unpacked_data[18:24])
                msg = Float64MultiArray()
                msg.data = data_array.tolist()
                self.publisher.publish(msg)
                self.get_logger().info(f'Received: {msg.data}')
            else:
                self.get_logger().warn(f'Incomplete data received: {len(data)} bytes')
        except socket.timeout:
            pass
        except struct.error as e:
            self.get_logger().error(f'Error unpacking data: {e}')
        except Exception as e:
            self.get_logger().error(f'Error receiving or processing data: {e}')

def main(args=None):
    rclpy.init(args=args)
    sender = FanucCmd()
    receiver = FanucPub()
    executor = rclpy.executors.MultiThreadedExecutor()
    
    executor.add_node(sender)
    executor.add_node(receiver)

    try:
        executor.spin()  # 允许 Ctrl+C 终止
    except KeyboardInterrupt:
        sender.get_logger().info("Shutting down due to keyboard interrupt...")
    finally:
        sender.destroy_node()
        receiver.destroy_node()
        executor.shutdown()  # 确保清理 executor
        rclpy.shutdown()


if __name__ == '__main__':
    main()
