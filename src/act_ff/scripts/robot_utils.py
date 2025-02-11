import rospy
import threading
import numpy as np
import time
from collections import deque
from std_msgs.msg import Float64MultiArray, Float64, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from constants import DT
from src.act_ff.scripts.fanuc_controller import FanucController

class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.serials = ['332322070892', "332522076772"]
        self.camera_names = ['gripper_top', 'top']
        self.lock = threading.Lock()

        if init_node:
            rospy.init_node('image_recorder', anonymous=True)

        # 存储图像数据
        self.images = {name: None for name in self.camera_names}
        self.timestamps = {name: None for name in self.camera_names}

        # 订阅 ROS 话题
        self.subscribers = {}
        for serial, cam_name in zip(self.serials, self.camera_names):
            topic = f'/camera_{cam_name}/image_raw'
            self.subscribers[cam_name] = rospy.Subscriber(topic, Image, lambda data, name=cam_name: self.image_cb(name, data))

        # 记录时间戳（调试用）
        if self.is_debug:
            self.debug_timestamps = {name: deque(maxlen=50) for name in self.camera_names}

        # 启动监听线程
        self.running = True
        self.thread = threading.Thread(target=self.listen_loop)
        self.thread.daemon = True
        self.thread.start()

    def image_cb(self, cam_name, data):
        """图像回调函数，更新最新图像数据"""
        with self.lock:
            self.images[cam_name] = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            self.timestamps[cam_name] = (data.header.stamp.secs, data.header.stamp.nsecs)
        if self.is_debug:
            self.debug_timestamps[cam_name].append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)

    def listen_loop(self):
        """ROS监听循环，保持 `subscriber` 持续运行"""
        rate = rospy.Rate(20)  # 20Hz 更新
        try:
            while self.running and not rospy.is_shutdown():
                rate.sleep()
        except KeyboardInterrupt:
            print(f"\n[INFO] {self.__class__.__name__} stopped by Ctrl+C.")
            self.running = False

    def get_images(self):
        """返回最新的图像数据"""
        with self.lock:
            return {name: self.images[name] for name in self.camera_names}

    def stop(self):
        """停止监听线程"""
        self.running = False
        self.thread.join()

    def print_diagnostics(self):
        """打印相机的采样频率"""
        def dt_helper(timestamps):
            ts = np.array(timestamps)
            return np.mean(ts[1:] - ts[:-1]) if len(ts) > 1 else 0

        for cam_name in self.camera_names:
            if len(self.debug_timestamps[cam_name]) > 1:
                image_freq = 1 / dt_helper(self.debug_timestamps[cam_name])
                print(f"{cam_name}: {image_freq:.2f} Hz")
        print()

class Recorder:
    def __init__(self, init_node=True, is_debug=False):
        self.is_debug = is_debug
        self.lock = threading.Lock()

        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_force = None
        self.gripper_command = None
        self.gripper_pos = None

        if init_node:
            rospy.init_node('recorder', anonymous=True)
        self.fanuc_controller=FanucController()

        # 订阅 ROS 话题
        rospy.Subscriber("/robot_state", Float64MultiArray, self.puppet_state_cb)
        rospy.Subscriber("/robot_cmd", Float64MultiArray, self.puppet_arm_commands_cb)
        rospy.Subscriber("/gripper_force", Int32, self.puppet_gripper_force_cb)
        rospy.Subscriber("/gripper_pos", Float64, self.puppet_gripper_pos_cb)

        # 调试用时间戳
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)

        # 启动监听线程
        self.running = True
        self.thread = threading.Thread(target=self.listen_loop)
        self.thread.daemon = True
        self.thread.start()

    def puppet_state_cb(self, data):
        """订阅机器人状态"""
        with self.lock:
            self.qpos = data.data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def puppet_arm_commands_cb(self, data):
        """订阅手臂命令"""
        with self.lock:
            self.fanuc_controller.send_udp_data(data.data)
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def puppet_gripper_force_cb(self, data):
        """订阅夹爪力"""
        with self.lock:
            self.gripper_force = data.data

    def puppet_gripper_pos_cb(self, data):
        """订阅夹爪位置"""
        with self.lock:
            self.gripper_pos=data.data

    def listen_loop(self):
        rate = rospy.Rate(50)  # 50Hz 更新
        try:
            while self.running and not rospy.is_shutdown():
                rate.sleep()
        except KeyboardInterrupt:
            print(f"\n[INFO] {self.__class__.__name__} stopped by Ctrl+C.")
            self.running = False

    def stop(self):
        """停止监听线程"""
        self.running = False
        self.thread.join()

    def print_diagnostics(self):
        """打印状态更新频率"""
        def dt_helper(timestamps):
            ts = np.array(timestamps)
            return np.mean(ts[1:] - ts[:-1]) if len(ts) > 1 else 0

        if self.is_debug:
            if len(self.joint_timestamps) > 1:
                joint_freq = 1 / dt_helper(self.joint_timestamps)
                print(f'Joint States: {joint_freq:.2f} Hz')

            if len(self.arm_command_timestamps) > 1:
                arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
                print(f'Arm Commands: {arm_command_freq:.2f} Hz')

            if len(self.gripper_command_timestamps) > 1:
                gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)
                print(f'Gripper Commands: {gripper_command_freq:.2f} Hz')

            print()

def get_arm_joint_positions(bot):
    return bot.arm.core.joint_states.position[:6]

def get_arm_gripper_positions(bot):
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position

def setup_puppet_bot(bot):
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    bot.dxl.robot_set_operating_modes("group", "arm", "position")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)

def setup_master_bot(bot):
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)

def set_standard_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 800)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def set_low_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 100)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)
    bot.dxl.robot_torque_enable("single", "gripper", False)

def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)
    bot.dxl.robot_torque_enable("single", "gripper", True)

def main():
    rospy.init_node("robot_controller", anonymous=True)
    recorder = Recorder(init_node=False)  # 这里设为 False
    pub = rospy.Publisher('/robot_cmd', Float64MultiArray, queue_size=10)
    rospy.sleep(0.1)  # 确保 publisher 注册成功

    msg = Float64MultiArray()
    msg.data = [0, 0, 0, 0, -1.57, 0]

    rate = rospy.Rate(50)  # 10 Hz
    while not rospy.is_shutdown():
        pub.publish(msg)
        rospy.loginfo(f"Sent command: {msg.data}")
        rate.sleep()

if __name__ == "__main__":
    main()