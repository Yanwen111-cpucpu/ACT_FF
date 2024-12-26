import h5py
import matplotlib.pyplot as plt

# 替换为你的 HDF5 文件路径
file_path = "D:\I_love_study\Berkeley_Robot\HK\ACT_FF\data\sim_telepolicy\\validation.hdf5"

# 打开 HDF5 文件并读取 qpos 数据
with h5py.File(file_path, 'r') as f:
    # 检查 qpos 数据的位置
    if '/observations/qpos' in f:
        qpos_data = f['/observations/qpos'][:]
    else:
        raise KeyError("qpos dataset not found in HDF5 file.")

# 检查 qpos 数据形状
print("qpos data shape:", qpos_data.shape)

# 绘制前7行 qpos 数据，显示所有行
plt.figure(figsize=(12, 8))
for i in range(7):
    #plt.plot(range(qpos_data.shape[0]), qpos_data[:,i], label=f"Row {i+1}")
    plt.plot(range(250), qpos_data[:250,i], label=f"Row {i+1}")

plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("qpos Values", fontsize=12)
plt.title("First 7 Rows of qpos Data(Val)", fontsize=14)
plt.legend(fontsize=10)
plt.grid()
plt.show()