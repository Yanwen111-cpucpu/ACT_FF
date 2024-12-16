import os

#file_path = "D:/I_love_study/Berkeley_Robot/HK/ACT_FF/assets/fanuc_inner/CAD/base.stl"
file_path = "gripper_2/visual/sensor_hand.stl"
# 检查文件是否存在以及是否有读取权限
if os.path.exists(file_path):
    if os.access(file_path, os.R_OK):
        print(f"{file_path} 有读取权限。")
    else:
        print(f"{file_path} 没有读取权限。")
else:
    print(f"{file_path} 文件不存在。")
