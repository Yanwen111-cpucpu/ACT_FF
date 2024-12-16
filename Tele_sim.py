from assets.fanuc_inner import Single_Arm_Test_Pygame
import motor_control_sim
from queue import Queue, Empty
import threading
import time

gripper_pos_queue=motor_control_sim.gripper_pos_queue
feedback_queue=motor_control_sim.feedback_queue

#这里启动了control线程，根据仿真给电机传输力矩控制信息，并将电机的位置信息放入gripper_pos_queue
motor_thread=threading.Thread(target=motor_control_sim.main)
motor_thread.start()

 #这里启动了仿真线程，持续更新feedback_queue
sim_thread=threading.Thread(target=Single_Arm_Test_Pygame.load_env,args=(gripper_pos_queue,feedback_queue))
sim_thread.start()

try:
    while True:
        time.sleep(0.1)  # 主线程保持活动，捕获 Ctrl+C
except KeyboardInterrupt:
    print("Main thread received KeyboardInterrupt, stopping...")
    motor_control_sim.running = False  # 停止 motor_control 线程
    sim_thread.join()           # 确保仿真线程正确退出
    motor_thread.join()        # 确保 motor_control 线程正确退出
