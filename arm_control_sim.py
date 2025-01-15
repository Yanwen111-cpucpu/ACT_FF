from logging import exception
import os
import numpy as np
from dynamixel_sdk import *

class DXL_Arm():
    def __init__(self):

        self.connection=True

        if os.name == 'nt':
            import msvcrt
            def getch():
                return msvcrt.getch().decode()
        else:
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            def getch():
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return ch

         # Uses Dynamixel SDK library

        #********* DYNAMIXEL Model definition *********
        #***** (Use only one definition at a time) *****
        self.MY_DXL = 'X_SERIES'       # X330 (5.0 V recommended), X430, X540, 2X430
        # self.MY_DXL = 'MX_SERIES'    # MX series with 2.0 firmware update.
        # self.MY_DXL = 'PRO_SERIES'   # H54, H42, M54, M42, L54, L42
        # self.MY_DXL = 'PRO_A_SERIES' # PRO series with (A) firmware update.
        # self.MY_DXL = 'P_SERIES'     # PH54, PH42, PM54
        # self.MY_DXL = 'XL320'        # [WARNING] Operating Voltage : 7.4V


        # Control table address
        if self.MY_DXL == 'X_SERIES' or self.MY_DXL == 'MX_SERIES':
            ADDR_TORQUE_ENABLE          = 64
            self.ADDR_PRESENT_POSITION       = 132
            BAUDRATE                    = 57600
        elif self.MY_DXL == 'PRO_SERIES':
            ADDR_TORQUE_ENABLE          = 562       # Control table address is different in DYNAMIXEL model
            self.ADDR_PRESENT_POSITION       = 611
            BAUDRATE                    = 57600
        elif self.MY_DXL == 'P_SERIES' or self.MY_DXL == 'PRO_A_SERIES':
            ADDR_TORQUE_ENABLE          = 512        # Control table address is different in DYNAMIXEL model
            self.ADDR_PRESENT_POSITION       = 580
            BAUDRATE                    = 57600
        elif self.MY_DXL == 'XL320':
            ADDR_TORQUE_ENABLE          = 24
            self.ADDR_PRESENT_POSITION       = 37
            BAUDRATE                    = 1000000   # Default Baudrate of XL-320 is 1Mbps

        # DYNAMIXEL Protocol Version (1.0 / 2.0)
        # https://emanual.robotis.com/docs/en/dxl/protocol2/
        PROTOCOL_VERSION            = 2.0

        # Factory default ID of all DYNAMIXEL is 1
        DXL_ID                      = 5

        # Use the actual port assigned to the U2D2.
        # ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
        DEVICENAME                  = 'COM10'

        TORQUE_ENABLE               = 1     # Value for enabling the torque
        TORQUE_DISABLE              = 0     # Value for disabling the torque
        DXL_MOVING_STATUS_THRESHOLD = 20    # Dynamixel moving status threshold

        index = 0

        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.portHandler = PortHandler(DEVICENAME)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        try:
        # Open port
            if self.portHandler.openPort():
                print("Succeeded to open the port")
            else:
                print("Failed to open the port")
                print("Press any key to terminate...")
                getch()
                quit()


            # Set port baudrate
            if self.portHandler.setBaudRate(BAUDRATE):
                print("Succeeded to change the baudrate")
            else:
                print("Failed to change the baudrate")
                print("Press any key to terminate...")
                getch()
                quit()
        except Exception as e:
            self.connection=False
            print(f"Error during serial connection: {e}")
            return
            
        # Disable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        self.init_angle=self.get_joint_angle()
        
        # For EE
        self.ee_id = 7
        # 设置工作模式为 Current-Based Position Control
        self.torque_ee_disable()
        self.set_ee_mode(5)

        # 设置最大电流为 50mA（可调整）
        self.set_ee_max_current(30)
        self.enable_ee_torque()
        self.set_ee_goal_position(360)
        while (abs(self.get_ee_pos()-360)>1):
            time.sleep(0.2)
        print("EE motor is set")

    def get_joint_angle(self):
            # Read present position
        motor_angles=np.zeros(6)
        # motor_angles[0]=1.57
        if self.connection== True:
            for motor_id in range(1,7):
                if (self.MY_DXL == 'XL320'): # XL320 uses 2 byte Position Data, Check the size of data in your DYNAMIXEL's control table
                    dxl_present_position, _, _ = self.packetHandler.read2ByteTxRx(self.portHandler, motor_id+1, self.ADDR_PRESENT_POSITION )
                else:
                    dxl_present_position, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, self.ADDR_PRESENT_POSITION)

                motor_angle=dxl_present_position/4096*360 #编码器值转化角度值
                if motor_id == 1:
                    motor_angle = -(motor_angle-270)
                elif motor_id == 2:
                    motor_angle = (motor_angle-180)
                elif motor_id == 3:
                    motor_angle = (motor_angle-180)
                elif motor_id == 4:
                    motor_angle = (motor_angle-180)
                elif motor_id == 5:
                    motor_angle = -(motor_angle - 180)
                elif motor_id == 6:
                    motor_angle = -(motor_angle-270)
                motor_angles[motor_id-1]=motor_angle/180*3.14

            return motor_angles
        else:
            return motor_angles #np.zeros(6)

    def enable_ee_torque(self):
        """
        启用电机扭矩
        """
        ADDR_TORQUE_ENABLE = 64
        TORQUE_ENABLE = 1
        result, error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.ee_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to enable torque: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error enabling torque: {self.packetHandler.getRxPacketError(error)}")
        print("EE Torque enabled.")

    def torque_ee_disable(self):
        ADDR_TORQUE_ENABLE = 64
        TORQUE_DISABLE = 0
        # 1. 禁用扭矩
        result, error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.ee_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to disable torque: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Torque disable error: {self.packetHandler.getRxPacketError(error)}")
        #print("Torque disabled.")

    def set_ee_mode(self, mode):
        """
        设置电机工作模式
        :param mode: 模式值 (例如 5 为 Current-Based Position Control)
        """
        ADDR_OPERATING_MODE = 11



        # 2. 设置工作模式
        result, error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.ee_id, ADDR_OPERATING_MODE, mode
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to set mode: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error setting mode: {self.packetHandler.getRxPacketError(error)}")
        print(f"Operating mode set to {mode}.")


    def set_ee_max_current(self, current_ma):
        """
        设置电机的最大电流
        """
        ADDR_CURRENT_LIMIT = 38
        current_limit = int(current_ma / 2.69)  # 单位转换：1 = 2.69mA
        result, error = self.packetHandler.write2ByteTxRx(
            self.portHandler, self.ee_id, ADDR_CURRENT_LIMIT, current_limit
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to set max current: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error setting max current: {self.packetHandler.getRxPacketError(error)}")
        print(f"Max current set to {current_ma} mA.")

    def set_ee_goal_position(self, position_deg):
        """
        设置电机目标位置
        """
        ADDR_GOAL_POSITION = 116
        goal_position = int((position_deg / 360) * 4095)  # 将角度转换为位置值
        result, error = self.packetHandler.write4ByteTxRx(
            self.portHandler, self.ee_id, ADDR_GOAL_POSITION, goal_position
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to set goal position: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error setting goal position: {self.packetHandler.getRxPacketError(error)}")
        print(f"Goal position set to {position_deg} degrees")
        return goal_position

    def get_ee_pos(self):
        """
        获取当前电机位置
        """
        ADDR_PRESENT_POSITION = 132
        position, result, error = self.packetHandler.read4ByteTxRx(
            self.portHandler, self.ee_id, ADDR_PRESENT_POSITION
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to get position: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error getting position: {self.packetHandler.getRxPacketError(error)}")
        position= position/4095*360

        
        return position  

    def send_ee_force(self):
        self.enable_ee_torque()
        self.set_ee_goal_position(360)


    def stop(self):
        # Close port 
        self.portHandler.closePort()
        print('Arm Stopped')

def main():
    arm=DXL_Arm()
    print(arm.get_joint_angle()/3.14*180)

if __name__ == '__main__':
    main()