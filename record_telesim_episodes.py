import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import signal
import sys
import pygame

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy,TelePolicy
from datetime import datetime

import IPython
e = IPython.embed

# 捕获 Ctrl+C 信号




def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """



    task_name = args['task_name']

    dataset_dir = args['dataset_dir']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_dir= os.path.join(dataset_dir, current_time)
    
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name = 'top'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_telepolicy':
        policy_cls = TelePolicy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):

        pygame.init()
        # 设置窗口尺寸和MuJoCo模型
        width, height = 1279, 959
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MuJoCo Simulation Display")
        clock = pygame.time.Clock()

        episode_start=time.time()
        print(f'{episode_idx=}')
        print('Rollout out Telepolicy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        if policy_cls == TelePolicy:
            def signal_handler(sig, frame):
                print("Ctrl+C detected. Stopping TelePolicy and motor control...")
                policy_cls.stop()  # 调用 TelePolicy 的 stop 方法停止 motor control 线程
                print("TelePolicy and motor control stopped. Exiting program.")
                sys.exit(0)
            signal.signal(signal.SIGINT, signal_handler)
            policy = policy_cls(env,inject_noise)
        else:
            policy = policy_cls(inject_noise)
        # setup plotting
        if onscreen_render:
            plt_img = ts.observation['images'][render_cam_name]
        episode_result = None
        for step in range(episode_len):
            #policy = policy_cls(env,inject_noise)
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        policy.stop()
                        break  # 跳出事件循环
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_a:
                            # 检测到 'a'，成功终止
                            episode_result = 'success'
                            policy.stop()
                            break
                        elif event.key == pygame.K_d:
                            # 检测到 'd'，失败终止
                            episode_result = 'fail'
                            policy.stop()
                            break
                if episode_result is not None:
                    break
                plt_img1 = ts.observation['images'][render_cam_name]
                plt_img2 = ts.observation['images']['front_close']   # 第二个摄像头
                plt_img3 = ts.observation['images']['angle']
                plt_img4 = ts.observation['images']['gripper_top']

                img_surface1 = pygame.surfarray.make_surface(np.flipud(np.rot90(plt_img1)))
                img_surface2 = pygame.surfarray.make_surface(np.flipud(np.rot90(plt_img2)))
                img_surface2 = pygame.transform.scale(img_surface2, img_surface1.get_size())
                img_surface3 = pygame.surfarray.make_surface(np.flipud(np.rot90(plt_img3)))
                img_surface3 = pygame.transform.scale(img_surface3, img_surface1.get_size())
                img_surface4 = pygame.surfarray.make_surface(np.flipud(np.rot90(plt_img4)))
                img_surface4 = pygame.transform.scale(img_surface4, img_surface1.get_size())
                #print(f"qpos:{ts.observation['qpos'][6]}")
                # 显示图像
                screen.blit(img_surface1, (0, 0))
                screen.blit(img_surface2, (plt_img1.shape[1], 0))  # 右侧显示
                screen.blit(img_surface3,(0,plt_img1.shape[0])) # 下左方显示
                screen.blit(img_surface4,(plt_img3.shape[1],plt_img1.shape[0])) # 下右方显示
                pygame.display.flip()
                #print(f'ts_obs:{ts.observation["c_force"]}')
                # 控制帧率
                clock.tick(60)  # 可调整帧率
        if episode_result is None:
            episode_result = 'fail'
                

        pygame.quit()
        # episode_return = np.sum([ts.reward for ts in episode[1:]])
        # episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        # if episode_max_reward == env.task.max_reward:
        #     print(f"{episode_idx=} Successful, {episode_return=}")
        # else:
        #     print(f"{episode_idx=} Failed")
        if episode_result == 'success':
            print(f"{episode_idx=} Successful,time={time.time()-episode_start}")
            success.append(1)
        else:
            print(f"{episode_idx=} Failed,time={time.time()-episode_start}")
            success.append(0)
            continue #Not record when fail

        joint_traj = [ts.observation['arm_gripper_ctrl'][:7].copy() for ts in episode]

        #print(joint_traj.pop(-1))
        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

        # clear unused variables
        del env
        del policy

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (7,)         'float64'
        force                   (1,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/force': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # # truncate here to be consistent
        # joint_traj = joint_traj[:-1]
        # episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1


        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/observations/force'].append(ts.observation['c_force'])
            data_dict['/action'].append(action)

            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            force = obs.create_dataset('force',(max_timesteps,1))
            action = root.create_dataset('action', (max_timesteps,7))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        time.sleep(1)

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

