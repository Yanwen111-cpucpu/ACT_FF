import sys
import os
import glfw
import numpy as np
import pygame
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

# 加载 XML 模型
XML_DIR = "./assets"
XML_PATH = os.path.join(XML_DIR, "bimanual_viperx_transfer_cube.xml")

class TransferCubeTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        # action仅包含单臂6自由度和夹爪左右指的1个信号，共7个元素

        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = None
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        return 0
class MujocoViewer:
    def __init__(self, xml_path):
        self.physics = mujoco.Physics.from_xml_path(xml_path)
        self.task = TransferCubeTask(random=False)
        self.env = control.Environment(self.physics, self.task, time_limit=20, control_timestep=0.02,
                                       n_sub_steps=None, flat_observation=False)
        # 初始化 Pygame
        pygame.init()
        self.width, self.height = 640, 480
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("MuJoCo Viewer")
        self.clock = pygame.time.Clock()
    
    def render(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            self.env.step([])  # 这里可以传入控制信号
            img = self.physics.render(height=self.height, width=self.width, camera_id='top')
            img = np.flipud(img)  # 翻转图像以适应 Pygame
            surf = pygame.surfarray.make_surface(img)
            self.screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    viewer = MujocoViewer(XML_PATH)
    try:
        viewer.render()
    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        viewer.close()
