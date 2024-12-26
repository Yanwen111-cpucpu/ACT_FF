from msilib.schema import Class
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS, START_ARM_POSE
from ee_sim_env import make_ee_sim_env
import threading
import dxl_motor_control_sim
import arm_control_sim
import signal
import time
import asyncio
import threading

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        # if self.right_trajectory[0]['t'] == self.step_count:
        #     self.curr_right_waypoint = self.right_trajectory.pop(0)
        # next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)#虽然只定义了几个时点，但可以插值生成整个路径
        # right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            #right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        #action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, np.zeros(8)])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        # self.right_trajectory = [
        #     {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
        #     {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
        #     {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
        #     {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
        #     {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
        #     {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
        #     {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
        #     {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
        #     {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        # ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]

class TelePolicy:
    def __init__(self,env,inject_noise=False):
        self.motor= dxl_motor_control_sim.DynamixelMotor() #末端夹爪控制
        self.arm=arm_control_sim.DXL_Arm()
        self.env=env
        # self.gripper_pos_queue = motor_control_sim.gripper_pos_queue
        # self.feedback_queue = motor_control_sim.feedback_queue
        self.step_count = 0
        self.inject_noise = inject_noise
        self.running = True  # 添加一个运行标志
        self.time=time.time()
        #self.motor.send_force(0.1)  #初始化时先给一个力矩命令，后面等抓到物体了再更新，避免一直发送指令增大延迟（0.1s)
        self.action=np.zeros(14)
        self.left_joint_angles=np.zeros(14)
        self.left_gripper=0
        self.force_feedback=0.1
        self.force_feedback_bool=False

        self.loop=None
        self.thread = threading.Thread(target=self.run_event_loop, daemon=True)
        self.thread.start()
        self.tasks=[]

        while self.loop is None:
            time.sleep(0.1)
        
        # 启动异步任务
        self.loop.call_soon_threadsafe(asyncio.create_task, self.start_async_tasks())

    def stop(self):
        # 停止 motor control 线程
        if self.loop and not self.loop.is_closed():
            for task in self.tasks:
                task.cancel()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()  # 等待线程结束
            self.loop.close()
        self.running = False
        self.motor.stop()
        self.arm.stop()
    def __call__(self, ts):
        time_start=self.time
        if not self.running:
            return np.zeros(14)
        # 生成动作
        action = self.action
        #print(f'gripper signal sent:{action[6]}')
        # 添加噪声
        
        if self.inject_noise:
            action += np.random.uniform(-0.01, 0.01, action.shape)

        self.step_count += 1
        
        signal.signal(signal.SIGINT, self.signal_handler)#检测ctrl+C打断，安全退出
        time_end=time.time()
        #print(f'time:{time_end-time_start}')
        self.time=time_end
        return action
    def run_event_loop(self):
        # 确保每个线程有独立的事件循环
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop()
        try:
            if not self.loop.is_running():
                self.loop.run_forever()
        finally:
            self.loop.close()
    async def update_joint_angles(self):
        while self.running:
            try:
                self.left_joint_angles = self.arm.get_joint_angle()-self.arm.init_angle+ START_ARM_POSE[:6]#每次初始化时位于START_ARM_POSE
                await asyncio.sleep(0.06)
            except asyncio.CancelledError:
                break  # 任务被取消
            except Exception as e:
                print(f"Error in update_joint_angles: {e}")
                break
    async def update_gripper_position(self):
        """异步更新夹爪位置"""
        while self.running:
            try:
                if self.motor is not None:
                    position=self.motor.get_pos()
                    position_map=max(0,min((360-position)/15,4)/114.29) #映射到slave端，0-3.5cm工作范围 
                    self.left_gripper = position_map
                    #print(f'left_finger_ctrl:{self.left_gripper}')
                    #print(self.left_gripper)
                else:
                    self.left_gripper +=0.005
                #print(f"[GRIPPER_POS] {self.left_gripper}")
                await asyncio.sleep(0.06)
            except asyncio.CancelledError:
                break  # 任务被取消
            except Exception as e:
                print(f"Error in update_joint_angles: {e}")
                break
    async def send_force_feedback(self):
        """异步发送力反馈信号"""
        while self.running:
            try:
                #print('enter send force')
                act_pos = self.env.physics.named.data.qpos['gripper_left_finger']

                if self.contact(self.env.physics):
                    self.force_feedback = float(0.3 + (self.left_gripper - act_pos) * 20)
                    self.force_feedback_bool=True
                else:
                    self.force_feedback = 0.3  # 无接触，给一个复位力矩
                    self.force_feedback_bool=False
                    self.motor.torque_disable()
                if self.force_feedback_bool:
                    self.motor.send_force()
                #print(f"[FORCE_FEEDBACK] {self.force_feedback}")
                await asyncio.sleep(0.06)
            except asyncio.CancelledError:
                break  # 任务被取消
            except Exception as e:
                print(f"Error in update_joint_angles: {e}")
                break
        
    async def generate_action(self):
        while self.running:
            try:
                self.action[:7] = np.concatenate([self.left_joint_angles, [self.left_gripper]])
                await asyncio.sleep(0.02)
            except asyncio.CancelledError:
                break  # 任务被取消
            except Exception as e:
                print(f"Error in update_joint_angles: {e}")
                break
    async def start_async_tasks(self):
        """启动所有异步任务"""
        self.tasks.append(asyncio.create_task(self.update_joint_angles()))
        self.tasks.append(asyncio.create_task(self.update_gripper_position()))
        self.tasks.append(asyncio.create_task(self.send_force_feedback()))
        self.tasks.append(asyncio.create_task(self.generate_action()))

    # def generate_action(self):
    #     # 动作维度为每个手臂的 6 个关节角度 + 夹爪开合程度 = 7 个元素
    #     #这里获取成员变量的最新值，将用于更新数据的函数都独立异步运行，降低延迟

    #     default_position = np.zeros(14)  # 左右手臂各 7 个元素
    #     time_start=self.time
    #     if self.motor.ser is not None:
    #         gripper_pos = self.motor.get_pos()
    #     else:
    #         gripper_pos = None

    #     left_joint_angles = self.arm.get_joint_angle() #如果没有连接机械臂，返回np.zeros(6)

    #     action= default_position

    #     if gripper_pos is not None:
    #         left_gripper = gripper_pos  # 左夹爪位移控制信号，右夹爪为相反数

    #     else:
    #         left_gripper=0 #没连电机时测试用
        
    #     action[:7] = np.concatenate([left_joint_angles, [left_gripper]])

    #     #更新force_feedback
    #     act_pos=self.env.physics.named.data.qpos['gripper_left_finger']

    #     if self.contact(self.env.physics):
    #         force_value=float(0.1+(left_gripper-act_pos)*50) #接触后线性增大力反馈
    #     else:
    #         force_value=0.1  # 无接触，给一个复位力矩
    #     self.motor.send_force(force_value)
    #     time_end=time.time()
    #     print(f'time:{time_end-time_start}')
    #     self.time=time_end
    #     return action

    def contact(self, physics):
        """检测左右夹爪是否接触物体"""
        touch_left_gripper = False
        touch_right_gripper = False

        # 如果没有任何接触点，直接返回 False
        if physics.data.ncon == 0:
            print("[DEBUG] No contacts detected")
            return False

        for i_contact in range(physics.data.ncon):
            try:
                # 检查索引有效性，访问接触点
                contact = physics.data.contact[i_contact]
                id_geom_1 = contact.geom1
                id_geom_2 = contact.geom2
                name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
                name_geom_2 = physics.model.id2name(id_geom_2, 'geom')

                # 检测左夹爪接触
                if name_geom_1 == "finger_left" or name_geom_2 == "finger_left":
                    touch_left_gripper = True
                    if contact.efc_address >= 0:  # 确保接触点有效
                        normal_force = physics.data.efc_force[contact.efc_address]
                        print(f"[DEBUG] Normal force at contact {i_contact}: {normal_force}")
                    return True

            except IndexError as e:
                # 捕获潜在的索引错误
                print(f"[ERROR] Contact index out of range: {e}")
                return False


            # 如果遍历完成没有检测到接触，返回 False
            #print("[DEBUG] No contact detected between grippers and objects")
            return False

    def signal_handler(self,sig, frame):
        print("Stopping TelePolicy...")
        self.stop()  # 调用 stop 以安全停止 motor
        exit(0)


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

