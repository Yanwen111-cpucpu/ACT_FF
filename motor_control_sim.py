'''
用于对力反馈电机进行控制，接收sensor回传的正应力，并将电机编码器的角度信息通过UDP传输给夹爪进行position control
电机的返回信息形如t064601FFFF7FF81D，其中t0646为标识符，不变，01为电机ID，FFFF为电机位置（这里超过了一圈，恒为FFFF）
后面是速度和力矩信息，详见中空系列电机控制使用说明
Run this before motor_command.py to start the server first.
'''

from concurrent.futures import thread
from operator import imod
from tkinter import TRUE
import serial
import time
import socket
import threading
from queue import Queue
import signal

# gripper_pos_queue = Queue()
# feedback_queue=Queue()

class MotorControl():
    def __init__(self):
        super().__init__()
        try:
            self.ser = init_serial('COM9')
            print("Motor Serial connection established.")
            self.run()
        except Exception as e:
            self.ser = None
            print("Warning: Motor Serial connection failed.")
        self.Pmax=1 #电机位置参数
        self.command='t00188000846000000900\r'
        self.response=''
    def stop(self):
        send_command(self.ser, 't0018FFFFFFFFFFFFFFFd\r')  # 发送停止命令
        print('Motor stopped')
        self.ser.close()

    def run(self):
        # 发送 S8\r 命令，设置CAN波特率为1M
        response_s8 = send_command(self.ser,'S8\r')
        #print(f"Response to S8 command: {response_s8}")
        
        response_init= send_command(self.ser,'t0018FFFFFFFFFFFFFFFE\r') #Set zero position    

        # 发送 Y5\r 命令
        response_y5 = send_command(self.ser,'Y5\r')
        #print(f"Response to Y5 command: {response_y5}")

        # 发送 O\r\n 命令，打开电机
        response_o = send_command(self.ser,'O\r\n')
        #print(f"Response to O command: {response_o}")

        for _ in range(2):
            response_d = send_command(self.ser,'t0018000800000000800\r')
            #print(f"Response to t0018000800000000800 command: {response_d}")

        response_mode=send_command(self.ser,'t0018FFFFFFFFFFFFFFFB\r') #switch to position-velocity-torque control mode

        response_loop= send_command(self.ser,'t001880B6823000000A00\r')#NOTICE there should be 16 bits information besides t0018

        response_init= send_command(self.ser,'t0018FFFFFFFFFFFFFFFC\r') #initiallize the motor

    def send_force(self, force_value):
        if self.ser== None:
            return
        command='t00188000846000000900\r'#the command repeatedly sent to the motor
        # if not self.feedback_queue.empty():
        #     feedback =force_feedback2str(self.feedback_queue)  # 获取队列中的第一个反馈信息（float）
        #     #print(feedback)
        #     # 替换命令中的最后三个字符
        #     command = command[:18] + feedback + '\r'
        #     #print(f'Sending modified command: {command}')
        feedback = force_feedback2str(force_value)
        self.command = command[:18]+feedback+'\r'
        self.response=send_command(self.ser,self.command)
        #print("force sent")

    def get_pos(self):

        response_loop= self.response
        if response_loop == '':
            return 0
        #print(f'Response to speed control command:{response_loop}')
        code=int(response_loop[7:11],16)
        position=((code-0x8000)/0x8000)*360*self.Pmax  #假设操作范围为0-60°
        position_map=max(0,min(position/15,4)/114.29) #映射到slave端，0-3.5cm工作范围 
        #print(f'Position:{response_loop}')
        # self.gripper_pos_queue.put(position_map)
        # print(f'position:{position}')
        return position_map



def init_serial(port):
    # 打开串口连接
    ser = serial.Serial(
    port=port,        # 这里替换为实际的串口号（如 'COM3'，'/dev/ttyUSB0' 等）
    baudrate=1e6,      # 设置合适的波特率 (与电机控制器通讯速率一致)
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1           # 超时设为1秒
    )
    return ser

def send_command(ser,command):
    """
    发送命令并等待响应
    """
    ser.write(command.encode())  # 将命令转换为字节并发送
    time.sleep(0.02)              # 等待一段时间，确保命令被处理
    response = ser.read_all().decode('utf-8')  # 读取串口返回的所有内容
    return response

def force_feedback2str(force_value):

    # try:
    #     while True:
            # data = 0 #TODO
            # if not data:
            #     break  # 如果没有接收到数据，断开循环
            # decoded_data = data.decode('utf-8')  # 将字节串解码为字符串            
            # # 从接收到的字符串中提取反馈值（去掉 'Message ' 部分）
            # if 'Message ' in decoded_data:
            #     feedback_value = float(decoded_data.split('Message ')[-1].strip())  # 提取数值部分
            #     feedback_value=round(feedback_value,3)
            #     print(f"Feedback value: {feedback_value}")
    #feedback_value=feedback_queue.get_nowait()
    feedback_code=hex(round((force_value/4*0x800)+0x800))

    feedback_code_str=feedback_code[2:]
    #print(f'feedback_code:{feedback_code_str}')
    return feedback_code_str
    # except Exception as e:
    #     print(f"Error in force_feedback: {e}")



'''
def motor_control():
    Pmax=1 #电机位置参数

    ser=init_serial('COM9')

    # 发送 S8\r 命令，设置CAN波特率为1M
    response_s8 = send_command(ser,'S8\r')
    #print(f"Response to S8 command: {response_s8}")
    
    response_init= send_command(ser,'t0018FFFFFFFFFFFFFFFE\r') #Set zero position    

    # 发送 Y5\r 命令
    response_y5 = send_command(ser,'Y5\r')
    #print(f"Response to Y5 command: {response_y5}")

    # 发送 O\r\n 命令，打开电机
    response_o = send_command(ser,'O\r\n')
    #print(f"Response to O command: {response_o}")
    


    for _ in range(2):
        response_d = send_command(ser,'t0018000800000000800\r')
        #print(f"Response to t0018000800000000800 command: {response_d}")

    response_mode=send_command(ser,'t0018FFFFFFFFFFFFFFFB\r') #switch to position-velocity-torque control mode

    response_loop= send_command(ser,'t001880B6823000000A00\r')#NOTICE there should be 16 bits information besides t0018

    response_init= send_command(ser,'t0018FFFFFFFFFFFFFFFC\r') #initiallize the motor

    command='t00188000846000000900\r'#the command repeatedly sent to the motor

    
    # 启动 force_feedback 线程
    # feedback_thread = threading.Thread(target=force_feedback, args=(feedback_queue))
    # feedback_thread.start()
    try:
        while(running):
            #response_mode=send_command('t0018FFFFFFFFFFFFFFFA\r') #switch to velocity-torque control mode
            if not feedback_queue.empty():
                feedback =force_feedback2str(feedback_queue)  # 获取队列中的第一个反馈信息（float）
                #print(feedback)
                # 替换命令中的最后三个字符
                command = command[:18] + feedback + '\r'
                #print(f'Sending modified command: {command}')
            response_loop= send_command(ser,command)
            #print(f'Response to speed control command:{response_loop}')
            code=int(response_loop[7:11],16)
            position=((code-0x8000)/0x8000)*360*Pmax  #假设操作范围为0-60°
            position_map=max(0,min(position/15,4)/200) #映射到slave端，0-2cm工作范围
            #print(f'Position:{position_map}')
            gripper_pos_queue.put(position_map)
            print(f'position:{position}')
    except KeyboardInterrupt:
        send_command(ser,'t0018FFFFFFFFFFFFFFFd\r')# Stop the motor
        ser.close()
        running=False
        print('exit')
    finally:
        send_command(ser,'t0018FFFFFFFFFFFFFFFd\r')# Stop the motor
        ser.close()
        running=False
        print('exit')
'''

def main():
    # 创建 MotorControlThread 实例并启动线程
    motor_thread = MotorControl()
    
    # 定义信号处理函数，捕获 Ctrl+C 并停止 motor_thread
    def signal_handler(sig, frame):
        print("Ctrl+C detected. Stopping motor control...")
        motor_thread.stop()  # 停止 motor control 线程
        print("Motor control stopped. Exiting program.")
        exit(0)

    # 设置信号处理程序
    signal.signal(signal.SIGINT, signal_handler)

    # 保持主线程运行
    try:
        while True:
            #motor_thread.join(1)  # 主线程保持运行，但定期检查 motor_thread 的状态
            motor_thread.send_force(1)
            motor_thread.get_pos()
            time.sleep(0.1)
    except KeyboardInterrupt:
        # 备用的退出处理，确保在捕获到 Ctrl+C 时线程安全退出
        signal_handler(None, None)


if __name__ == "__main__":
    main()
