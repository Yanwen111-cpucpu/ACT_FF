import h5py
import matplotlib.pyplot as plt
import cv2

# 替换为你的 HDF5 文件路径
file_path = "D:\I_love_study\Berkeley_Robot\HK\ACT_FF\data\sim_telepolicy\episode_2.hdf5"
# 打开 HDF5 文件并读取 force 数据
with h5py.File(file_path, 'r') as f: