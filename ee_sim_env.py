from pickle import TRUE
import numpy as np
import collections
import os

from constants import DT, PUPPET_GRIPPER_POSITION_OPEN, XML_DIR
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_box_color, sample_poses, sample_box_size, sample_insertion_pose, update_mocap_pos_relative_to_gripper
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed


def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_telepolicy' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TeleTask(random=False)
        env = control.Environment(physics, task, time_limit=120, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)

    else:
        raise NotImplementedError
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl])) 
        # a 2-finger gripper needs 2 independent signal to control the position of each part.

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = physics.data.ctrl[:16].copy()

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        # np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        # np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # # right
        # np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
        # np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        open_gripper_control = np.array([0,0])
        np.copyto(physics.data.ctrl[6:8], open_gripper_control)
        #np.copyto(physics.data.ctrl[14:],open_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640,camera_id='front_close')
        obs['images']['gripper_top']= physics.render(height=480, width=640, camera_id='gripper_top')
        obs['images']['front_close']= physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        # obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        # obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used as input of encoder
        obs['arm_gripper_ctrl'] = physics.data.ctrl.copy()
        obs['c_force']=np.maximum(0,(obs['arm_gripper_ctrl'][6]-physics.named.data.qpos['gripper_left_finger'].copy())*100)
        #print(obs['c_force'])
        return obs

    def get_reward(self, physics):
        raise NotImplementedError

class TeleTask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def before_step(self, action,physics):
        # 假设 action 包含左右手臂各 6 个关节角度和 1 个夹爪开合度，共计 14 个元素
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # np.copyto(physics.data.qpos[:6], action_left[:6])
        np.copyto(physics.data.ctrl[:6], action_left[:6])
        physics.data.ctrl[1]=action_left[1] #必须有这行，否则joint2的运动就很奇怪，我也不知道为什么
        #physics.data.qpos[1]=action_left[0]
        # 设置夹爪开合度
        #g_left_ctrl,g_right_ctrl = update_mocap_pos_relative_to_gripper(physics,action_left[6])
        g_left_ctrl = action_left[6]
        #g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[6])
        np.copyto(physics.data.ctrl[6:8], np.array([g_left_ctrl, -g_left_ctrl])) 
        # np.copyto(physics.data.mocap_pos[0], g_left_ctrl) mocap绑定手指会影响整个机械臂的运动
        # np.copyto(physics.data.mocap_pos[1], g_right_ctrl)
        # 可视化接触点

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        poses = sample_poses()
        cube_size = sample_box_size()
        cube_color = sample_box_color()
        box_joint_start_idx = physics.model.name2id('red_box_joint', 'joint')
        container_start_idx = physics.model.name2id('target_container','body')
        box_start_idx=physics.model.name2id('red_box','geom')
        np.copyto(physics.data.qpos[box_joint_start_idx : box_joint_start_idx + 7], poses['box_pose'])
        np.copyto(physics.model.body_pos[container_start_idx : container_start_idx + 3], poses['container_pose'][:3])
        np.copyto(physics.model.geom_size[box_start_idx], cube_size)
        np.copyto(physics.model.geom_rgba[box_start_idx], cube_color)
        # np.copyto(physics.model.geom_rgba[box_start_idx : box_start_idx + 4], cube_color)
        # print(f"randomized cube position to {cube_position}")
        # link_6_start_idx=physics.model.name2id('joint_6','joint')
        # gripper_start_idx=physics.model.name2id('gripper_root','joint')
        # np.copyto(physics.data.qpos[gripper_start_idx:gripper_start_idx+3],physics.data.xpos[link_6_start_idx])
        # np.copyto(physics.data.qpos[gripper_start_idx+3:gripper_start_idx+7],physics.data.xquat[link_6_start_idx])
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
            
        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward

