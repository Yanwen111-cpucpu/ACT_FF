# gripper_controller.py (ROS 1 Version based on Manual)
import rospy
import minimalmodbus
import serial
from std_msgs.msg import Float64, Int32

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
        
        rospy.Timer(rospy.Duration(0.02), self.check_gripper_status)
        rospy.Timer(rospy.Duration(0.02), self.read_pos)  # 🔥 让 read_pos() 也定期运行

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

    def control_gripper(self, position):
        position = position.data
        if not (0 <= position<= 0.025):
            print('Gripper position out of range (0-25 mm)')
            return
        try:
            self.instrument.write_registers(0x0002, [int(position * 1000), 0])
            print(f'Set gripper position to {position} mm')
        except Exception as e:
            print(f'Error setting gripper position: {e}')

    def read_pos(self):
        pos_data = self.instrument.read_registers(0x0042, 2)
        pos_msg = Float64(data=pos_data[0] / 1000.0)
        self.position_publisher.publish(pos_msg)

    def check_gripper_status(self, event): #force
        try:
            status = self.instrument.read_register(0x0041, 0)
            gripper_state = 1 if status == 2 else 0
            self.state_publisher.publish(Int32(data=gripper_state*3.5))
        except Exception as e:
            rospy.logerr(f'Error reading gripper state: {e}')

if __name__ == '__main__':
    try:
        GripperController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
