# fanuc_controller.py (ROS 1 Version)
import rospy
import socket
import struct
import numpy as np
from std_msgs.msg import Float64MultiArray

class FanucCmd:
    """ 订阅 /robot_cmd 并通过 UDP 发送给 Fanuc 机器人 """
    
    def __init__(self):
        rospy.init_node('fanuc_sender', anonymous=True)
        
        self.target_ip = rospy.get_param('~target_ip', '192.168.1.100')
        self.target_port = rospy.get_param('~target_port', 3827)
        self.frequency = rospy.get_param('~frequency', 50)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.subscriber = rospy.Subscriber('/robot_cmd', Float64MultiArray, self.send_udp_data)
        rospy.loginfo(f'Listening for commands on /robot_cmd')
    
    def send_udp_data(self, msg):
        if len(msg.data) != 6:
            rospy.logwarn('Received incorrect data length')
            return
        try:
            msg_array = np.array(msg.data) / 3.14 * 180  # 角度转换
            packed_data = struct.pack('<6d', *msg_array)
            self.sock.sendto(packed_data, (self.target_ip, self.target_port))
            rospy.loginfo(f'Sent: {msg_array.tolist()}')
        except Exception as e:
            rospy.logerr(f'Error sending data: {e}')


class FanucPub:
    """ 监听 UDP 并发布 /robot_state 话题 """
    
    def __init__(self):
        rospy.init_node('fanuc_receiver', anonymous=True)
        
        self.local_ip = rospy.get_param('~local_ip', '0.0.0.0')
        self.port = rospy.get_param('~port', 9600)
        self.frequency = rospy.get_param('~frequency', 50)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_ip, self.port))
        self.sock.settimeout(0.01)
        
        self.publisher = rospy.Publisher('/robot_state', Float64MultiArray, queue_size=10)
        self.rate = rospy.Rate(self.frequency)
        
        rospy.loginfo(f'Listening for UDP messages on {self.local_ip}:{self.port}')
    
    def receive_udp_message(self):
        while not rospy.is_shutdown():
            try:
                data, _ = self.sock.recvfrom(36 * 8)
                if len(data) == 36 * 8:
                    unpacked_data = struct.unpack('<36d', data)
                    data_array = np.array(unpacked_data[18:24])
                    msg = Float64MultiArray()
                    msg.data = data_array.tolist()
                    self.publisher.publish(msg)
                    rospy.loginfo(f'Received: {msg.data}')
                else:
                    rospy.logwarn(f'Incomplete data received: {len(data)} bytes')
            except socket.timeout:
                pass
            except struct.error as e:
                rospy.logerr(f'Error unpacking data: {e}')
            except Exception as e:
                rospy.logerr(f'Error receiving or processing data: {e}')
            self.rate.sleep()


def main():
    sender = FanucCmd()
    receiver = FanucPub()
    
    try:
        receiver.receive_udp_message()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
