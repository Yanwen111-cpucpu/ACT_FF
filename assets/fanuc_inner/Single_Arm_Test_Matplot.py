from dm_control import mujoco
from dm_control import suite
from dm_control import viewer
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pygame

def run_simulation(physics):
    while True:
        physics.step()  # 执行一步仿真
        # 渲染图像并在查看器中显示
        pixels = physics.render(height=480, width=640, camera_id=0)

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



def render_multiple_cameras(physics, camera_ids):
    # 初始化画布
    num_columns = 2
    num_rows = int(np.ceil(len(camera_ids) / num_columns))
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 9))
    plt.ion()  # 开启交互模式

    # 设置子图
    ims = []
    for i, ax in enumerate(axes.flat):
        if i < len(camera_ids):
            # 初始化每个相机视图
            pixels = physics.render(height=480, width=640, camera_id=camera_ids[i])
            im = ax.imshow(pixels)
            ims.append(im)
        ax.axis('off')  # 隐藏坐标轴

    plt.tight_layout()

    # 每个时间步更新视图
    time_step = 0
    while True:
        # 应用控制并推进仿真
        apply_control(physics, time_step)
        physics.step()  # 重要！执行时间步以应用控制
        
        # 渲染图像
        images = [physics.render(height=480, width=640, camera_id=camera_id) for camera_id in camera_ids]

        # 更新每个子图的内容
        for im, img in zip(ims, images):
            im.set_data(img)

        plt.pause(0.001)  # 暂停一小段时间以更新显示内容
        plt.draw()
        
        time_step += 1  # 增加时间步

    plt.ioff()  # 关闭交互模式
    plt.show()


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

# 自定义关节控制函数
def apply_control(physics, timestep):
    target_qpos = -1.57  # 目标角度
    kp = physics.model.actuator_gainprm[4][0]#在XML文件中已有kp的控制
    kd = 3   # 微分增益
    
    # 当前关节位置和速度
    qpos = physics.data.qpos[4]
    qvel = physics.data.qvel[4]

    # 计算 PD 控制力矩
    torque = kp * (target_qpos - qpos) - kd * qvel #不可以直接对qpos进行修改，这样没有改变系统的动力学，仿真会在下一帧按原先的动力学重新计算，所以会“回弹”
    
    # 施加控制信号（力矩）
    physics.data.ctrl[4] = torque

camera_ids=["left_pillar","right_pillar","top","angle","front_close","left_cam_focus"]
render_multiple_cameras(physics,camera_ids)

# # 遍历并打印关节名称和对应的 DOF 索引
# for i in range(physics.model.njnt):
#     # 获取关节的ID并使用 physics.model.id2name 来获取关节名称
#     joint_name = physics.model.id2name(i, 'joint')
#     dof_index = physics.model.jnt_dofadr[i]
#     print(f"Joint '{joint_name}' is associated with DOF index {dof_index}.")

# # 主循环
# running = True
# time_step=0
# while running:

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # 控制第一个关节
#     apply_control(physics, time_step)

#     # 执行仿真步骤
#     physics.step()
#     #images = render_multiple_cameras(physics, camera_ids)
#     pixels = physics.render(height=480, width=640, camera_id=2)
#     # 将第一个相机的图像转换为 pygame 表面并显示
#     img_surface = pygame.surfarray.make_surface(np.flipud(pixels).swapaxes(0, 1))
#     screen.blit(pygame.transform.scale(img_surface, (1280, 720)), (0, 0))
#     pygame.display.flip()

#     time_step += 0.01  # 更新时间步长
#     time.sleep(0.01)  # 控制帧率

# # 退出
# pygame.quit()
# #-----------------------------------------------

# # 手动渲染循环
# for _ in range(100):
#     physics.step()  # 执行仿真步骤
#     render_multiple_cameras(physics,camera_ids)