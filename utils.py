import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
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
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


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
    container_x_range = [-0.6, 0.6]
    container_y_range = [0.5, 0.9]
    container_z_range = [0, 0]  # Container is on the ground

    # Box pose range
    box_x_range = [-0.6, 0.6]
    box_y_range = [0.5, 0.8]
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