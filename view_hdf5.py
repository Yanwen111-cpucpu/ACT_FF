import h5py
import matplotlib.pyplot as plt
import cv2

# 替换为你的 HDF5 文件路径
file_path = "data/sim_tele_cube/2024-12-06_17-32-57/episode_0.hdf5"
# 打开 HDF5 文件并读取 force 数据
with h5py.File(file_path, 'r') as f:
    # 确认 force 数据的位置
    # if 'observations/force' in f:
    #     force_data = f['/observations/qpos'][:]
    # else:
    #     raise KeyError("Force dataset not found in HDF5 file.")
        # 遍历帧并显示
    video_data=f['observations/images/gripper_top']
    for i, frame in enumerate(video_data):
        cv2.imshow("Frame", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 每帧显示30ms，按 'q' 退出q
            break

# # 绘制 force 数据
# plt.figure(figsize=(10, 6))
# plt.plot(force_data, label="Force", color="blue")
# plt.xlabel("Timestep")
# plt.ylabel("Force")
# plt.title("Force Over Time")
# plt.legend()
# plt.grid()
# plt.show()