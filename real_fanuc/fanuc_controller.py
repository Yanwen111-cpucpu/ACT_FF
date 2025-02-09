# fanuc_controller.py (ROS 1 Version)
import rospy
import socket
import struct
import numpy as np
from std_msgs.msg import Float64MultiArray
import atexit

class FanucController:
    
    def __init__(self):
        
        # 获取 ROS 参数，带默认值，避免获取失败
        self.target_ip = '192.168.1.100'
        self.target_port = 3827
        self.frequency = 50

        # 初始化 UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        

        atexit.register(self.cleanup)  # 兼容非 ROS 退出

    def send_udp_data(self,data):
        """ 处理 ROS 订阅消息，并通过 UDP 发送 """
        if len(data) != 6:
            print('Received incorrect data length')
            return
        try:
            # **确保数据类型正确**
            msg_array = np.array(data, dtype=np.float64) / 3.14 * 180  # 角度转换
            
            # **确保 struct.pack 数据匹配**
            packed_data = struct.pack('<6d', *msg_array)
            
            # **UDP 发送**
            self.sock.sendto(packed_data, (self.target_ip, self.target_port))
        except Exception as e:
            rospy.logerr(f'Error sending data: {e}')

    def cleanup(self):
        """ 退出时关闭 socket """
        if hasattr(self, 'sock'):
            self.sock.close()



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
        
        self.publisher = rospy.Publisher('/robot_state', Float64MultiArray, queue_size=10) #degree or radius?
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
    sender = FanucController()
    receiver = FanucPub()
    
    try:
        receiver.receive_udp_message()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
