import numpy as np
import time
from constants import DT
from gripper_controller import GripperController
from fanuc_controller import FanucCmd, FanucPub
from std_msgs.msg import Float64MultiArray, Float64, Int32

import IPython
e = IPython.embed

class ImageRecorder:
    def __init__(self, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.serials= ['332322070892',"332522076772"]
        self.camera_names=['gripper_top','top']

        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for serial in self.serials:
            if serial =="332322070892":
                cam_name='gripper_top'
            elif serial =="332522076772":
                cam_name='top'
            else:
                raise NotImplementedError
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            if cam_name == 'cam_wrist':
                callback_func = self.image_cb_cam_wrist
            elif cam_name == 'cam_top':
                callback_func = self.image_cb_cam_top
            else:
                raise NotImplementedError
            rospy.Subscriber(f'/camera_{cam_name}/image_raw', Image, callback_func)
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.secs * 1e-9)

    def image_cb_cam_wrist(self, data):
        cam_name = 'gripper_top'
        return self.image_cb(cam_name, data)

    def image_cb_cam_top(self, data):
        cam_name = 'top'
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()

class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        from collections import deque
        import rospy
        from sensor_msgs.msg import JointState
        from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_force = None
        self.gripper_command = None
        self.gripper_state = None
        self.is_debug = is_debug

        #待启动：FanucPub(robot_state)、FanucCmd(robot_cmd#TODO)、gripper_controller(gripper_state、gripper_pos)、gripper_cmd#TODO、camera_pub

        if init_node:
            rospy.init_node('recorder', anonymous=True)
        rospy.Subscriber(f"/robot_state'", Float64MultiArray, self.puppet_state_cb)
        rospy.Subscriber(f"/robot_cmd", JointGroupCommand, self.puppet_arm_commands_cb) #TODO
        rospy.Subscriber(f"/gripper_state", Int32, self.puppet_gripper_force_cb)
        rospy.Subscriber(f"/gripper_state", Float64, self.puppet_gripper_state_cb)
        rospy.Subscriber(f"/gripper_cmd", JointSingleCommand, self.puppet_gripper_commands_cb) #TODO

        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def puppet_state_cb(self, data):
        self.qpos = data
        #self.qvel = data.velocity
        #self.effort = data.effort
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def puppet_arm_commands_cb(self, data):
        self.arm_command = data
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def puppet_gripper_force_cb(self, data):
        self.gripper_force = data

    def puppet_gripper_state_cb(self, data):
        self.gripper_state = data

    def puppet_gripper_commands_cb(self, data):
        self.gripper_command = data
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f'{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n')

def get_arm_joint_positions(bot):
    return bot.arm.core.joint_states.position[:6]

def get_arm_gripper_positions(bot):
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position

def move_arms(bot_list, target_pose_list, move_time=1):
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)

def move_grippers(bot_list, target_pose_list, move_time):
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)

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