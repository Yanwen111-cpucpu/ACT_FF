from dynamixel_sdk import *  # Dynamixel SDK
import time

class DynamixelMotor:
    def __init__(self, port, baudrate=57600, motor_id=1):
        """
        初始化 Dynamixel 电机类
        """
        self.port = port
        self.baudrate = baudrate
        self.motor_id = motor_id

        # 初始化 PortHandler 和 PacketHandler
        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(2.0)  # 使用协议版本 2.0

        # 打开串口
        if self.port_handler.openPort() and self.port_handler.setBaudRate(self.baudrate):
            print(f"Dynamixel Motor {self.motor_id} initialized on {self.port}.")
        else:
            raise Exception("Failed to connect to Dynamixel motor.")

    def enable_torque(self):
        """
        启用电机扭矩
        """
        ADDR_TORQUE_ENABLE = 64
        TORQUE_ENABLE = 1
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to enable torque: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error enabling torque: {self.packet_handler.getRxPacketError(error)}")
        print("Torque enabled.")

    def set_mode(self, mode):
        """
        设置电机工作模式
        :param mode: 模式值 (例如 5 为 Current-Based Position Control)
        """
        ADDR_OPERATING_MODE = 11
        ADDR_TORQUE_ENABLE = 64
        TORQUE_DISABLE = 0

        # 1. 禁用扭矩
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to disable torque: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Torque disable error: {self.packet_handler.getRxPacketError(error)}")
        print("Torque disabled.")

        # 2. 设置工作模式
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.motor_id, ADDR_OPERATING_MODE, mode
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to set mode: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error setting mode: {self.packet_handler.getRxPacketError(error)}")
        print(f"Operating mode set to {mode}.")


    def set_max_current(self, current_ma):
        """
        设置电机的最大电流
        """
        ADDR_CURRENT_LIMIT = 38
        current_limit = int(current_ma / 2.69)  # 单位转换：1 = 2.69mA
        result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler, self.motor_id, ADDR_CURRENT_LIMIT, current_limit
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to set max current: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error setting max current: {self.packet_handler.getRxPacketError(error)}")
        print(f"Max current set to {current_ma} mA.")

    def set_goal_position(self, position_deg):
        """
        设置电机目标位置
        """
        ADDR_GOAL_POSITION = 116
        goal_position = int((position_deg / 360) * 4095)  # 将角度转换为位置值
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, self.motor_id, ADDR_GOAL_POSITION, goal_position
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to set goal position: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error setting goal position: {self.packet_handler.getRxPacketError(error)}")
        print(f"Goal position set to {position_deg} degrees")

    def get_position(self):
        """
        获取当前电机位置
        """
        ADDR_PRESENT_POSITION = 132
        position, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, self.motor_id, ADDR_PRESENT_POSITION
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to get position: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error getting position: {self.packet_handler.getRxPacketError(error)}")
        return (position / 4095) * 360  # 转换为角度值

    def stop_motor(self):
        """
        停止电机
        """
        ADDR_TORQUE_ENABLE = 64
        TORQUE_DISABLE = 0
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
        )
        if result != COMM_SUCCESS:
            raise Exception(f"Failed to stop motor: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise Exception(f"Error stopping motor: {self.packet_handler.getRxPacketError(error)}")
        print("Motor stopped.")
        self.port_handler.closePort()


def main():
    motor = DynamixelMotor(port='COM10', motor_id=1)

    position=motor.get_position()
    goal_position=360

    # # 启用扭矩
    # motor.enable_torque()

    # 设置工作模式为 Current-Based Position Control
    motor.set_mode(5)

    # 设置最大电流为 50mA（可调整）
    motor.set_max_current(30)

    motor.enable_torque()

    motor.set_goal_position(goal_position)

    #while(abs((position-goal_position)/goal_position)>=0.02):
    while True:
        try:

            # 设置目标位置为 360°

            # # 获取当前位置
            position = motor.get_position()
            motor.enable_torque()
            print(f"Current position: {position:.2f} degrees")
            motor.set_goal_position(goal_position)

            time.sleep(5)
            motor.set_mode(5)
            time.sleep(5)
        except KeyboardInterrupt:
            print("Ctrl+C detected. Stopping motor...")
            motor.stop_motor()
    motor.stop_motor()

if __name__ == "__main__":
    main()
