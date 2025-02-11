import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env
import rospy

from constants import DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
from src.act_ff.scripts.robot_utils import Recorder, ImageRecorder
from src.act_ff.scripts.robot_utils import setup_master_bot, setup_puppet_bot
from src.act_ff.scripts.fanuc_controller import FanucController,FanucPub
from src.act_ff.scripts.gripper_controller import GripperController
from src.act_ff.scripts.cam_pub import RealSenseCamera
from std_msgs.msg import Float64MultiArray, Float64, Int32

import IPython
e = IPython.embed

class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node):

        if init_node:
            rospy.init_node('real_env', anonymous=True)
        self.recorder= Recorder(init_node=False)
        self.image_recorder = ImageRecorder(init_node=False)
        
        self.gripper_cmd=rospy.Publisher('/gripper_cmd', Float64, queue_size=10)
        self.fanuc_cmd=rospy.Publisher('/robot_cmd', Float64MultiArray, queue_size=10)

        self.msg_fanuc=Float64MultiArray()
        self.msg_gripper=Float64()

    def get_qpos(self):
        left_qpos_raw = self.recorder.qpos
        left_arm_qpos = left_qpos_raw[:6]
        left_gripper_pos = self.recorder.gripper_pos
        return np.concatenate([left_arm_qpos, np.array([left_gripper_pos])])

    def get_qvel(self):
        left_qvel_raw = self.recorder_left.qvel
        right_qvel_raw = self.recorder_right.qvel
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    def get_effort(self):
        left_effort_raw = self.recorder_left.effort
        right_effort_raw = self.recorder_right.effort
        left_robot_effort = left_effort_raw[:7]
        right_robot_effort = right_effort_raw[:7]
        return np.concatenate([left_robot_effort, right_robot_effort])

    def get_force(self):
        return self.recorder.gripper_force

    def get_images(self):
        return self.image_recorder.get_images()

    def _reset_joints(self):
        reset_position = START_ARM_POSE[:6]
        self.msg_fanuc.data=reset_position
        self.fanuc_cmd.publish(self.msg_fanuc)

    def _reset_gripper(self):
        """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
        self.msg_gripper.data=0
        self.gripper_cmd.publish(self.msg_gripper)
        time.sleep(1)
        self.msg_gripper.data=0.025
        self.gripper_cmd.publish(self.msg_gripper)
        

    def get_observation(self):
        obs = collections.OrderedDict()
        qpos = self.get_qpos()
        obs['qpos'] = np.concatenate([qpos, np.zeros(7)])
        obs['qvel'] = None
        #obs['effort'] = self.get_effort()
        obs['c_force'] = np.array([self.get_force()])
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        
        self.msg_fanuc.data=action[:6]
        self.msg_gripper.data=action[-1]

        self.fanuc_cmd.publish(self.msg_fanuc)
        self.gripper_cmd.publish(self.msg_gripper)

        time.sleep(DT) #if too slow, get this bigger

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14) # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    action[7+6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action


def make_real_env(init_node):
    env = RealEnv(init_node)
    return env


def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.
    It first reads joint poses from both master arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_left_wrist'

    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()
