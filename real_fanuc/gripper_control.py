import minimalmodbus
import serial
import time

# 设备配置
device_address = 1  # 设备地址
port = 'COM9'  # 串口端口，需根据实际情况调整
baudrate = 115200  # 波特率

# 初始化 Modbus 设备
instrument = minimalmodbus.Instrument(port, device_address)
instrument.serial.baudrate = baudrate
instrument.serial.bytesize = 8
instrument.serial.parity = serial.PARITY_NONE
instrument.serial.stopbits = 1
instrument.serial.timeout = 1  # 通信超时（秒）
instrument.mode = minimalmodbus.MODE_RTU

# 控制夹爪电流的函数
def set_gripper_current(current):
    """
    设置夹爪的夹持电流。
    :param current: 夹持电流值，范围 0.1 ~ 0.5 A
    """
    if not (0.1 <= current <= 0.5):
        raise ValueError("电流值必须在 0.1 ~ 0.5 A 之间")

    # 夹持电流寄存器地址为 0x0006，功能码 0x10（写多个寄存器）
    register_address = 0x0006

    # 将浮点数电流值转换为 Modbus 格式
    current_value = minimalmodbus._float_to_bytes(current)

    try:
        instrument.write_float(register_address, current)
        print(f"成功设置夹持电流为 {current} A")
    except Exception as e:
        print(f"设置夹持电流失败: {e}")

def set_gripper_position(position):
    """
    设置夹爪的位置。
    :param position: 夹爪位置，范围 0 ~ 50 mm
    """
    if not (0 <= position <= 50):
        raise ValueError("位置值必须在 0 ~ 50 mm 之间")

    # 夹持位置寄存器地址为 0x0002
    register_address = 0x0002

    try:
        # 使用 write_float 写入浮点数数据（单位：mm）
        instrument.write_float(register_address, position)
        print(f"成功设置夹爪位置为 {position} mm")
    except Exception as e:
        print(f"设置夹爪位置失败: {e}")

# 示例：设置夹爪电流为 0.3 A
set_gripper_position(50)
time.sleep(5)
set_gripper_current(0.1)
set_gripper_position(0)
