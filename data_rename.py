import os
import shutil  # 用于移动文件

# 目标目录路径
root_dir = "D:\I_love_study\Berkeley_Robot\HK\ACT_FF\data\sim_tele_cube"  # 修改为你的根目录路径
target_dir = "D:\I_love_study\Berkeley_Robot\HK\ACT_FF\data\sim_telepolicy"

# 初始化全局计数器
episode_index = 0

# 遍历根目录下的所有文件夹
for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):  # 确保是文件夹
        # 遍历文件夹内的所有文件
        for file in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):  # 确保是文件
                # 获取文件后缀名
                file_ext = os.path.splitext(file)[-1]
                # 生成新的文件名
                new_file_name = f"episode_{episode_index}{file_ext}"
                new_file_path = os.path.join(target_dir, new_file_name)
                # 重命名并移动文件到根目录
                shutil.move(file_path, new_file_path)
                print(f"Moved and Renamed: {file_path} -> {new_file_path}")
                # 更新计数器
                episode_index += 1

print("文件重命名并移动完成！")
