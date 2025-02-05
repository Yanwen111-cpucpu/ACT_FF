from tracemalloc import start
from turtle import st
import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, delta_timestamps=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        try:
            with h5py.File(dataset_path, 'r') as root:
                is_sim = root.attrs['sim']
                original_action_shape = root['/action'].shape
                episode_len = 420 #hardcode
                #print(f"episode_len of {episode_id}:{episode_len}")
                if sample_full_episode:
                    start_ts = 0
                else:
                    start_ts = np.random.choice(original_action_shape[0])
                # get observation at start_ts only
                qpos = root['/observations/qpos'][start_ts]
                qvel = root['/observations/qvel'][start_ts]
                force = root['/observations/force'][start_ts]
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                    #print(root[f'/observations/images/{cam_name}'][start_ts])
                # get all actions after and including start_ts
                if is_sim:
                    action = root['/action'][start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
        except (OSError, KeyError) as e:
            # 捕获文件名不匹配或数据键缺失的错误
            print(f"Warning: Skipping episode {episode_id} due to error: {e}")

        self.is_sim = is_sim
        padded_action = np.zeros((episode_len,original_action_shape[1]), dtype=np.float32)
        padded_action[:len(action)] = action
        is_pad = np.zeros(episode_len)
        is_pad[len(action):] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        force_data = torch.from_numpy(force)
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        force_data = (force_data - self.norm_stats["force_mean"])/self.norm_stats["force_std"]
        # print(f"Sample {index}:")
        # print(f"  qpos shape: {qpos_data.shape}, type: {type(qpos_data)}")
        # print(f"  image shape: {image_data.shape}, type: {type(image_data)}")
        # print(f"  force shape: {force_data.shape}, type: {type(force_data)}")
        # print(f"  action shape: {action_data.shape}, type: {type(action_data)}")
        # print(f"  is_pad shape: {is_pad.shape}, type: {type(is_pad)}")
        
        return image_data, qpos_data, action_data,force_data,is_pad

class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats,delta_timestamps=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.delta_timestamps = delta_timestamps
        
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        try:
            with h5py.File(dataset_path, 'r') as root:
                is_sim = root.attrs['sim']
                original_action_len = root['/action'].shape[0]
                episode_len = 420 #hardcode
                #print(f"episode_len of {episode_id}:{episode_len}")
                if sample_full_episode:
                    start_ts = 0

                else:
                    start_ts = np.random.choice(original_action_len)

                action_length = 16  # action_length in delta_timestamps

                if start_ts == 0:
                    # 提取第 0 时刻的数据
                    qpos_base = root['/observations/qpos'][0:1]  # 第 0 时刻的数据，形状为 (1, features)
                    qpos = np.repeat(qpos_base, 2, axis=0)       # 重复 2 次，形状为 (2, features)

                    qvel_base = root['/observations/qvel'][0:1]
                    qvel = np.repeat(qvel_base, 2, axis=0)

                    force_base = root['/observations/force'][0:1]
                    force = np.repeat(force_base, 2, axis=0)

                    original_action_base = root['/action'][0:15]
                    original_action = np.vstack([root['/action'][0:1], original_action_base])

                    image_base_dict = dict()
                    image_dict = dict()
                    for cam_name in self.camera_names:
                        image_base_dict[cam_name] = root[f'/observations/images/{cam_name}'][0:1]
                        image_dict[cam_name] = np.repeat(image_base_dict[cam_name],2,axis=0)

                    is_pad = np.zeros(action_length, dtype=np.int32)
                    is_pad[0] = 1

                else:
                    qpos = root['/observations/qpos'][start_ts-1:start_ts+1] #hardcode
                    qvel = root['/observations/qvel'][start_ts-1:start_ts+1]
                    force = root['/observations/force'][start_ts-1:start_ts+1]
                
                    original_action_end = min(start_ts + 15, root['/action'].shape[0])  # 防止越界
                    # 提取合法范围的数据
                    original_action = root['/action'][max(0, start_ts - 1):original_action_end]

                    # 初始化 is_pad 数组，默认所有值为 0
                    
                    is_pad = np.zeros(action_length, dtype=np.int32)

                    # 计算越界部分的长度
                    valid_length = original_action.shape[0]
                    pad_length = action_length - valid_length  # 需要填充的数量

                    if pad_length > 0:
                        # 如果存在越界部分，填充最后一个合法数据
                        padding_data = np.repeat(original_action[-1:], pad_length, axis=0)
                        original_action = np.vstack((original_action, padding_data))

                        # 更新 is_pad，标记越界部分为 1
                        is_pad[-pad_length:] = 1

                    image_dict = dict()
                    for cam_name in self.camera_names:
                        image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts-1:start_ts+1]

        except (OSError, KeyError) as e:
            # 捕获文件名不匹配或数据键缺失的错误
            print(f"Warning: Skipping episode {episode_id} due to error: {e}")

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        original_action_data = torch.from_numpy(original_action).float()
        force_data = torch.from_numpy(force)
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k ... h w c -> ... k c h w', image_data) #3,n_steps,480,640,3 -> n_steps,3,3,480,640

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        force_data = (force_data - self.norm_stats["force_mean"])/self.norm_stats["force_std"]
        action_data = (original_action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            

        return image_data, qpos_data, action_data,force_data,is_pad

def padding(tensor, target_length):
    """
    裁剪序列到固定长度 target_length
    """
    current_length = tensor.shape[0]
    padding = torch.zeros(target_length - current_length, tensor.shape[1])
    return torch.cat((tensor, padding), dim=0)



def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    all_force_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        try:
            # 尝试打开文件
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                force = root['/observations/force'][()]
                action = root['/action'][()]
        except (OSError, KeyError) as e:
            # 捕获文件名不匹配或数据键缺失的错误，跳过当前循环
            print(f"Warning: Skipping episode {episode_idx} due to error: {e}")
            continue



        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_force_data.append(torch.from_numpy(force))

    if not all_qpos_data or not all_action_data or not all_force_data:
        raise ValueError("No valid data found. Please check the dataset directory or file names.")

    target_length = 500  # 假设episode_len=1000
    all_qpos_data = [padding(tensor, target_length) for tensor in all_qpos_data]
    all_action_data = [padding(tensor, target_length) for tensor in all_action_data]
    all_force_data = [padding(tensor, target_length) for tensor in all_force_data]


    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_force_data = torch.stack(all_force_data)

    # 创建掩码
    qpos_mask = (all_qpos_data != 0).float()
    action_mask = (all_action_data != 0).float()

    # 计算均值
    qpos_mean = (all_qpos_data * qpos_mask).sum(dim=[0, 1], keepdim=True) / qpos_mask.sum(dim=[0, 1], keepdim=True)
    action_mean = (all_action_data * action_mask).sum(dim=[0, 1], keepdim=True) / action_mask.sum(dim=[0, 1], keepdim=True)

    # 计算标准差
    qpos_var = ((all_qpos_data - qpos_mean) ** 2 * qpos_mask).sum(dim=[0, 1], keepdim=True) / qpos_mask.sum(dim=[0, 1], keepdim=True)
    qpos_std = torch.sqrt(qpos_var)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    action_var = ((all_action_data - action_mean) ** 2 * action_mask).sum(dim=[0, 1], keepdim=True) / action_mask.sum(dim=[0, 1], keepdim=True)
    action_std = torch.sqrt(action_var)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    # normalize force data
    force_mean = all_force_data.mean(dim=[0, 1], keepdim=True)
    force_std = all_force_data.std(dim=[0, 1], keepdim=True)
    force_std = torch.clip(force_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
        "force_mean": force_mean.numpy().squeeze(),
        "force_std": force_std.numpy().squeeze(),
    }

    return stats

def stack_camera_images(image_data):
    """
    将多个相机的图像张量堆叠为一个张量。

    Parameters:
    - image_data (dict): 包含多个相机图像数据的字典，值是 torch.Tensor。

    Returns:
    - torch.Tensor: 堆叠后的张量，形状为 (num_cameras, batch_size, channels, height, width)。
    """
    # 将所有相机的张量提取到列表中
    all_cam_images = [tensor for tensor in image_data.values()]

    # 确保所有张量的形状一致
    for i, tensor in enumerate(all_cam_images):
        if tensor.shape != all_cam_images[0].shape:
            raise ValueError(f"Tensor at index {i} has a different shape: {tensor.shape}")

    # 堆叠成一个张量
    stacked_images = torch.stack(all_cam_images, dim=0)  # 在第 0 维堆叠
    return stacked_images


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,delta_timestamps=None):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos, action and force
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    if delta_timestamps is None:
        train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats,delta_timestamps)
        val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats,delta_timestamps)
    else:
        train_dataset = DiffusionDataset(train_indices, dataset_dir, camera_names, norm_stats,delta_timestamps)
        val_dataset = DiffusionDataset(val_indices, dataset_dir, camera_names, norm_stats,delta_timestamps)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def load_previous_and_future_frames(
    item,
    dataset,
    episode_data_index,
    delta_timestamps
) :
    current_idx = item["index"]
    # get indices of the frames associated to the episode, and their timestamps
    ep_data_id_from = 0
    ep_data_id_to = dataset["actions"].shape[0]
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)

    for key in delta_timestamps:
        # get timestamps used as query to retrieve data of previous/future frames

        #print(f"key:{key}")
        delta_idx = delta_timestamps[key]
        query_idx = torch.tensor([current_idx + d for d in delta_idx])


        # TODO(rcadene): synchronize timestamps + interpolation if needed

        is_pad = (query_idx < ep_data_id_from) | (query_idx >= ep_data_id_to)

        # 将超出范围的索引限制在合法范围内
        query_idx_clamped = query_idx.clamp(ep_data_id_from, ep_data_id_to - 1)

        # 加载对应帧数据
        data_ids = ep_data_ids[query_idx_clamped]

        item[key] = dataset[key][data_ids]

        item[f"{key}_is_pad"] = is_pad

    return item

### env utils

def sample_box_size():

    x_range = [0.007, 0.017]
    y_range = [0.007, 0.017]
    z_range = [0.07, 0.2]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_size = np.random.uniform(ranges[:, 0], ranges[:, 1])

    return cube_size

def sample_box_color():
    r_range = [0, 1]
    g_range = [0, 1]
    b_range = [0, 1]
    a=1

    ranges = np.vstack([r_range, g_range,b_range])
    cube_color = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_color = np.concatenate([cube_color, np.array([a], dtype=np.float32)])

    return cube_color

def sample_poses():
    # Container pose range
    container_x_range = [0.6, 0.72]
    container_y_range = [0.45, 0.55]
    container_z_range = [0, 0]  # Container is on the ground

    # Box pose range
    box_x_range = [0.2, 0.3]
    box_y_range = [0.8, 0.92]
    box_z_range = [0.3, 0.3]  # Box starts above the ground

    # Generate container position
    container_ranges = np.vstack([container_x_range, container_y_range, container_z_range])
    container_position = np.random.uniform(container_ranges[:, 0], container_ranges[:, 1])
    container_quat = [1, 0, 0, 0]

    # Container bounds for checking if box is inside
    container_x_bounds = (container_position[0] - 0.3 / 2, container_position[0] + 0.3 / 2)
    container_y_bounds = (container_position[1] - 0.3 / 2, container_position[1] + 0.3 / 2)

    # Generate box position with check
    box_position = None
    while True:
        box_ranges = np.vstack([box_x_range, box_y_range, box_z_range])
        candidate_position = np.random.uniform(box_ranges[:, 0], box_ranges[:, 1])

        if not (container_x_bounds[0] <= candidate_position[0] <= container_x_bounds[1] and
                container_y_bounds[0] <= candidate_position[1] <= container_y_bounds[1]):
            box_position = candidate_position
            print(f'xbound:{container_x_bounds[0]},pos:{candidate_position[0]};ybound:{container_y_bounds[0]},pos:{candidate_position[1]}')
            break

    box_quat = [1, 0, 0, 0]

    return {
        "container_pose": np.concatenate([container_position, container_quat]),
        "box_pose": np.concatenate([box_position, box_quat])
    }



def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def update_mocap_pos_relative_to_gripper(physics, motor_signal):
    """
    更新基于夹爪坐标系的 Mocap body 位置，将其转换为世界坐标系下的值。

    Args:
        physics: MuJoCo 仿真环境
        motor_signal: 电机控制信号（相对位移，用于夹爪张合）
    """
    # 获取夹爪的全局位置和方向
    gripper_pos = physics.named.data.xpos['root']  # 夹爪中心的全局位置
    gripper_rot = physics.named.data.xmat['root'].reshape(3, 3)  # 夹爪的旋转矩阵（3x3）

    # 计算左手指和右手指的相对位移（在夹爪坐标系下）
    left_finger_offset = np.array([motor_signal-0.035, 0, 0]) 
    right_finger_offset = np.array([0.035-motor_signal, 0, 0]) 

    # 转换到世界坐标系
    left_finger_global = gripper_pos + gripper_rot @ left_finger_offset
    right_finger_global = gripper_pos + gripper_rot @ right_finger_offset

    return left_finger_global,right_finger_global
    # 更新 mocap body 的位置（在世界坐标系下）
    np.copyto(physics.data.mocap_pos[0], left_finger_global)  # 左手指的 Mocap body
    np.copyto(physics.data.mocap_pos[1], right_finger_global)  # 右手指的 Mocap body