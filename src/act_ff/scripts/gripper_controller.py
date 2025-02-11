import rospy
import minimalmodbus
import serial
import threading
from std_msgs.msg import Float64, Int32
import time
import struct

class GripperController:
    def __init__(self):
        rospy.init_node('gripper_controller', anonymous=True)

        self.state_publisher = rospy.Publisher('/gripper_force', Int32, queue_size=10)
        self.position_publisher = rospy.Publisher('/gripper_pos', Float64, queue_size=10)
        self.cmd_subscriber = rospy.Subscriber('/gripper_cmd', Float64, self.control_gripper)

        self.device_address = 1
        self.port = '/dev/ttyUSB0'  # Ubuntu 下的串口设备
        self.baudrate = 115200

        self.instrument = minimalmodbus.Instrument(self.port, self.device_address)
        self.instrument.serial.baudrate = self.baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity = serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout = 1
        self.instrument.mode = minimalmodbus.MODE_RTU

        self.lock = threading.Lock()  # 🔥 添加锁，防止并发访问

        rospy.Timer(rospy.Duration(0.02), self.check_gripper_status)
        rospy.Timer(rospy.Duration(0.02), self.read_pos)  # 🔥 让 read_pos() 也定期运行

        rospy.loginfo('Gripper controller node started.')
        self.initialize_gripper()

    def initialize_gripper(self):
        """初始化夹爪"""
        try:
            rospy.loginfo("Initializing gripper...")
            with self.lock:  
                self.instrument.write_register(0x0000, 1, functioncode=6)  # 发送初始化指令
            rospy.sleep(1)  # 等待初始化完成
            
            # 设定夹持速度 50mm/s
            with self.lock:
                self.instrument.write_registers(0x0004, [0x4248, 0x0000])
            rospy.sleep(0.5)

            # 设定夹持电流 0.3A
            with self.lock:
                self.instrument.write_registers(0x0006, [0x3E99, 0x999A])
            rospy.sleep(0.5)

            rospy.loginfo("Gripper initialized successfully.")

        except Exception as e:
            rospy.logerr(f'Error initializing gripper: {e}')

    def control_gripper(self, position):
        """控制夹爪开合"""
        position = position.data
        if not (0 <= position <= 0.025):  
            rospy.logwarn('Gripper position out of range (0-50 mm)')
            return
        try:
            with self.lock:
                # 计算 32-bit 位置值
                target_position = int(position * 1000*2)  # 例如 0.025m -> 25mm,single finger
                ieee754_bytes = struct.pack('>f', target_position)  # 转换成 IEEE 754 4 字节
                high_word, low_word = struct.unpack('>HH', ieee754_bytes)  # 拆分高16位和低16位
                
                # 发送目标位置（0x0002）
                self.instrument.write_registers(0x0002, [high_word, low_word])
            
            rospy.loginfo(f'Set gripper position to {position * 1000:.1f} mm')

        except Exception as e:
            rospy.logerr(f'Error setting gripper position: {e}')

    def read_pos(self, event):
        """读取夹爪当前位置"""
        try:
            with self.lock:
                pos_data = self.instrument.read_registers(0x0042, 2)
            raw_bytes = struct.pack('>HH', pos_data[0], pos_data[1])
            position = struct.unpack('>f', raw_bytes)[0]/1000/2
            pos_msg = Float64(data=position)
            self.position_publisher.publish(pos_msg)
            rospy.loginfo(f"Gripper pos: {pos_msg.data}")
        except Exception as e:
            rospy.logerr(f"Error reading gripper position: {e}")

    def check_gripper_status(self, event):
        """检查夹爪状态"""
        try:
            with self.lock:
                status = self.instrument.read_register(0x0041, 0)
            states = {0: "At position", 1: "Moving", 2: "Holding object", 3: "Dropped object"}
            rospy.loginfo(f"Gripper state: {states.get(status, 'Unknown')}")
            self.state_publisher.publish(Int32(data=status))
        except Exception as e:
            rospy.logerr(f'Error reading gripper state: {e}')

if __name__ == '__main__':
    try:
        GripperController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
