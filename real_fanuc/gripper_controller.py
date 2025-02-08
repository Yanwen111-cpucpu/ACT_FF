# gripper_controller.py (ROS 1 Version based on Manual)
import rospy
import minimalmodbus
import serial
from std_msgs.msg import Float64, Int32

class GripperController:
    def __init__(self):
        rospy.init_node('gripper_controller', anonymous=True)
        
        self.subscriber = rospy.Subscriber('/gripper_cmd', Float64, self.control_gripper)
        self.state_publisher = rospy.Publisher('/gripper_state', Int32, queue_size=10)
        self.position_publisher = rospy.Publisher('/gripper_pos', Float64, queue_size=10)

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
        
        rospy.Timer(rospy.Duration(0.3), self.check_gripper_status)
        rospy.loginfo('Gripper controller node started.')

        # 初始化夹爪
        self.initialize_gripper()

    def initialize_gripper(self):
        try:
            rospy.loginfo("Initializing gripper...")
            self.instrument.write_register(0x0000, 1, functioncode=6)
            rospy.sleep(2)
        except Exception as e:
            rospy.logerr(f'Error initializing gripper: {e}')

    def control_gripper(self, msg):
        position = msg.data
        if not (0 <= position <= 0.050):
            rospy.logwarn('Gripper position out of range (0-50 mm)')
            return
        try:
            self.instrument.write_registers(0x0002, [int(position * 1000), 0])
            rospy.loginfo(f'Set gripper position to {position} mm')
            pos_data = self.instrument.read_registers(0x0042, 2)
            pos_msg = Float64(data=pos_data[0] / 1000.0)
            self.position_publisher.publish(pos_msg)
        except Exception as e:
            rospy.logerr(f'Error setting gripper position: {e}')

    def check_gripper_status(self, event):
        try:
            status = self.instrument.read_register(0x0041, 0)
            gripper_state = 1 if status == 2 else 0
            self.state_publisher.publish(Int32(data=gripper_state))
        except Exception as e:
            rospy.logerr(f'Error reading gripper state: {e}')

if __name__ == '__main__':
    try:
        GripperController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
