from pickle import TRUE
import queue
from turtle import width
from dm_control import mujoco
from dm_control import suite
from dm_control import viewer
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pygame
import threading
from queue import Empty, Queue

# 自定义动作规范类，带有 minimum 和 shape 属性
class ActionSpec:
    def __init__(self, shape):
        self.shape = shape
        self.minimum = np.zeros(shape)
        self.maximum = np.zeros(shape)
        self.dtype=np.float32

# 定义一个简单的环境类
class SimpleEnvironment:
    def __init__(self, physics):
        self.physics = physics
    
    def action_spec(self):
        # 返回自定义的 ActionSpec 对象，满足 Viewer 的需求
        return ActionSpec(shape=(0,))  # 空动作规范，无需控制输入

    def reset(self):
        self.physics.reset()  # 重置仿真环境
        return self.physics
    
    def step(self, action):
        self.physics.step()  # 每次调用时执行一步仿真

# 自定义关节控制函数    
def apply_control(physics, gripper_pos,timestep):

    kp = physics.model.actuator_gainprm[7][0]*100 #在XML文件中已有kp的控制，调整该项、kd来做到比较好的遥操作效果
    kd = 1000  # 微分增益
    
    # 当前关节位置和速度
    #qpos_left = physics.data.qpos[3]
    #print(qpos_left)
    qpos_right=physics.data.qpos[7]
    #print(f"joint pos:{qpos}")
    #qvel_left = physics.data.qvel[3]
    qvel_right=physics.data.qvel[7]
    if not gripper_pos==None:
        target_qpos_left = gripper_pos  # 目标位置
        target_qpos_right = -gripper_pos
    else: #未连接motor时测试用
        # target_qpos_left=qpos_left
        # target_qpos_right=-qpos_right
        target_qpos_left=1.57
        #target_qpos_right=-timestep*100
        target_qpos_right=0.04
    # 计算 PD 控制力矩
    #torque_left = kp * (target_qpos_left - qpos_left) - kd * qvel_left #不可以直接对qpos进行修改，这样没有改变系统的动力学，仿真会在下一帧按原先的动力学重新计算，所以会“回弹”
    #torque_right=kp*(target_qpos_right-qpos_right)-kd*qvel_right
    damped_target_qpos_right=target_qpos_right-kd/kp*qvel_right
    #physics.data.ctrl[3] = torque_left
    physics.data.ctrl[7] = damped_target_qpos_right #这是位置控制模式
    
def run_pygame_render(physics, camera_ids,gripper_pos_queue,feedback_queue):
  # 初始化 Pygame
    pygame.init()
    num_cameras = len(camera_ids)
    
    # 定义每个摄像头的显示尺寸
    cam_width, cam_height = 320, 240  # 每个摄像头显示的宽和高
    rows = 2  # 显示两行
    cols = (num_cameras + 1) // rows  # 每行的列数
    screen_width = cam_width * cols
    screen_height = cam_height * rows
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('MuJoCo Simulation with Multiple Cameras')

    clock = pygame.time.Clock()
    timestep = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 应用控制信号
        try:
            # 使用 get 方法，并设置一个超时时间，以阻塞模式等待数据
            gripper_pos = gripper_pos_queue.get(timeout=0.1)  # 设置超时，以避免长时间阻塞
            print(f"gripper_pos:{gripper_pos}")
        except Empty:
            gripper_pos = None
            print("gripper_pos_queue is empty, skipping this timestep.")

        apply_control(physics, gripper_pos,timestep)
        if contact(physics):
            feedback_queue.put(2) #If contact, return a force feedback to the motor
        else:
            feedback_queue.put(0)
        # 渲染每个摄像头的图像并显示在屏幕上不同位置
        # for idx, camera_id in enumerate(camera_ids):
        #     # 获取摄像头的图像
        #     pixels = physics.render(height=cam_height, width=cam_width, camera_id=camera_id)
            
        #     # 转换为 Pygame 图像格式
        #     surf = pygame.surfarray.make_surface(np.flipud(pixels))

        #     # 计算在两行中的显示位置
        #     row, col = divmod(idx, cols)
        #     x = col * cam_width
        #     y = row * cam_height
        #     screen.blit(surf, (x, y))
        pixel=physics.render(height=cam_height*2,width=cam_width*2,camera_id=2)
        surf=pygame.surfarray.make_surface(np.flipud(pixel))
        screen.blit(surf,(0,0))

        pygame.display.flip()  # 更新整个窗口显示

        # 控制帧率
        clock.tick(60)  # 控制渲染速度为 60 FPS

        # 执行一步仿真
        physics.step()

        timestep += 0.002  # 增加时间步计数

    pygame.quit()

def contact(physics):
    all_contact_pairs=[]
    touch_left_gripper=False
    touch_right_gripper=False

    for i_contact in range(physics.data.ncon): #遍历所有接触对，转换成geom的name来表示
        id_geom_1 = physics.data.contact[i_contact].geom1
        id_geom_2 = physics.data.contact[i_contact].geom2
        name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
        name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)
    # 检查左夹爪和右夹爪是否与其他物体接触
        if name_geom_1 == "finger_left" or name_geom_2 == "finger_left":
            touch_left_gripper = True
        if name_geom_1 == "finger_right" or name_geom_2 == "finger_right":
            touch_right_gripper = True
    # 如果左夹爪和右夹爪都接触到物体，则返回 True
        if touch_left_gripper and touch_right_gripper:
            return True
    return False

def load_env(gripper_pos_queue, feedback_queue):
    os.chdir("D:/I_love_study/Berkeley_Robot/HK/ACT_FF/assets")

    # 设置自定义 MuJoCo 模型的路径
    model_path = 'bimanual_viperx_transfer_cube.xml'

    # 加载模型
    physics = mujoco.Physics.from_xml_path(model_path)


    # 实例化环境
    env = SimpleEnvironment(physics)
    #----------------------------------
    # 初始化 pygame
    # pygame.init()
    # screen = pygame.display.set_mode((800,600))
    # pygame.display.set_caption('Robot Simulation Control')



    camera_ids=["left_pillar","right_pillar","top","angle","front_close","left_cam_focus"]

    run_pygame_render(physics,camera_ids,gripper_pos_queue,feedback_queue)


if __name__ == '__main__':
    gripper_pos=Queue()
    feedback_queue=Queue()
    load_env(gripper_pos,feedback_queue)


    # # 遍历并打印关节名称和对应的 DOF 索引
    # for i in range(physics.model.njnt):
    #     # 获取关节的ID并使用 physics.model.id2name 来获取关节名称
    #     joint_name = physics.model.id2name(i, 'joint')
    #     dof_index = physics.model.jnt_dofadr[i]
    #     print(f"Joint '{joint_name}' is associated with DOF index {dof_index}.")


    # # 手动渲染循环
    # for _ in range(100):
    #     physics.step()  # 执行仿真步骤
    #     render_multiple_cameras(physics,camera_ids)

