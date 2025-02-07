import rclpy
from rclpy.node import Node
import socket
import struct
from std_msgs.msg import Float64MultiArray, Float64, Int32
import numpy as np
import minimalmodbus
import serial

class GripperController(Node):
    def __init__(self):
        super().__init__('gripper_controller')
        self.subscription = self.create_subscription(
            Float64, '/gripper_cmd', self.control_gripper, 10)
        self.publisher = self.create_publisher(Int32, '/gripper_state', 10)
        self.position_publisher = self.create_publisher(Float64, '/gripper_pos', 10)

        self.get_logger().info('Listening for gripper commands on /gripper_cmd')
        
        self.device_address = 1
        self.port = 'COM13'
        self.baudrate = 115200
        
        self.instrument = minimalmodbus.Instrument(self.port, self.device_address)
        self.instrument.serial.baudrate = self.baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity = serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout = 1
        self.instrument.mode = minimalmodbus.MODE_RTU
        
        self.timer = self.create_timer(0.3, self.check_gripper_status)  # 每0.3秒检测一次夹爪状态
    
    def control_gripper(self, msg):
        position = msg.data
        if not (0 <= position <= 0.025):
            self.get_logger().warn('Gripper position out of range (0-50 mm)')
            return
        try:
            self.instrument.write_float(0x0002, position*1000)
            self.get_logger().info(f'Set gripper position to {position} mm')
             # 读取夹爪位置
            position = self.instrument.read_float(0x0042)  # 读取夹爪当前位置
            pos_msg = Float64()
            pos_msg.data = position
            self.position_publisher.publish(pos_msg)
            self.get_logger().info(f'Gripper position: {position} mm')
        except Exception as e:
            self.get_logger().error(f'Error reading gripper data: {e}')
    
    def check_gripper_status(self):
        try:
            status = self.instrument.read_register(0x0041, 0)  # 读取夹持状态
            gripper_state = 1 if status == 2 else 0  # 2 = 夹持物体，其他 = 未夹持
            msg = Int32()
            msg.data = gripper_state
            self.publisher.publish(msg)
            self.get_logger().info(f'Gripper state: {gripper_state}')
        except Exception as e:
            self.get_logger().error(f'Error reading gripper state: {e}')
