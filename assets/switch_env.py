import pygame
import numpy as np
from dm_control import mujoco
import xml.etree.ElementTree as ET
import os

def modify_mujoco_xml(mode):
    """
    mode = 1: 无颜色、无纹理、无光影，黑色背景
    mode = 2: 有颜色、无纹理、无光影，灰色背景
    mode = 3: 有颜色、有纹理、有光影，灰色背景
    """
    xml_path = "assets/bimanual_viperx_transfer_cube.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 递归加载所有 include 文件
    #load_include_files(root)

    # 修改所有 geom 的颜色
    for geom in root.findall(".//geom"):
        if mode == 1:
            geom.set("rgba", "0.6 0.6 0.6 1")  # 设为灰色
            geom.attrib.pop("material", None)  # 移除材质

        elif mode == 2:
            if geom.get("name")=="table":
                geom.set("material", "tablecloth_plain_material")  
            else:
                geom.attrib.pop("material", None)  # 仅移除材质，不修改颜色
        elif mode == 3:
            geom.set("rgba","1 1 1 1")
            pass
    for light in root.findall(".//light"):
        if mode in [1, 2]:
            light.set("castshadow", "false")  # 关闭阴影
    # 保存修改后的 XML
    modified_xml_path = "assets/modified_env.xml"
    tree.write(modified_xml_path)

    return modified_xml_path

def render_simulation():
    """ 使用 pygame 渲染 MuJoCo 物理仿真 """
    pygame.init()

    # 设置窗口大小
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MuJoCo Simulation with Pygame")

    # 加载修改后的 MuJoCo XML
    mode = 3  # 选择模式 1, 2, 3
    modified_xml = modify_mujoco_xml(mode)
    physics = mujoco.Physics.from_xml_path(modified_xml)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 运行物理仿真
        physics.step()

        # 渲染 MuJoCo 场景到 numpy 数组
        img = physics.render(height, width, camera_id=2)
        img = np.flipud(img)  # 需要翻转以匹配 pygame 的坐标系

        # 转换 numpy 数组为 pygame 图像
        surf = pygame.surfarray.make_surface(img)

        # 显示到窗口
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # 控制帧率
        clock.tick(30)

    pygame.quit()

# 运行仿真
render_simulation()
