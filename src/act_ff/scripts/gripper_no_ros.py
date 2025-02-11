# gripper_test.py (Standalone Gripper Test based on Manual)
import minimalmodbus
import serial
import time
import struct

def test_gripper():
    device_address = 1
    port = '/dev/ttyUSB0'  # 确保你的设备路径正确
    baudrate = 115200
    
    # 初始化 Modbus 设备
    instrument = minimalmodbus.Instrument(port, device_address)
    instrument.serial.baudrate = baudrate
    instrument.serial.bytesize = 8
    instrument.serial.parity = serial.PARITY_NONE
    instrument.serial.stopbits = 1
    instrument.serial.timeout = 1
    instrument.mode = minimalmodbus.MODE_RTU
    
    try:
        print("Testing gripper: initializing, opening and closing...")
        
        # 发送初始化命令
        instrument.write_register(0x0000, 1, functioncode=6)
        time.sleep(2)  # 等待初始化完成
        
        # 设定夹持速度 50mm/s
        instrument.write_registers(0x0004, [0x4248, 0x0000])
        time.sleep(0.5)
        
        # 设定夹持电流 0.3A
        instrument.write_registers(0x0006, [0x3E99, 0x999A])
        time.sleep(0.5)
        
        # 设定夹爪打开（最大位置 50mm）
        instrument.write_registers(0x0002, [0x4248, 0x0000])
        time.sleep(1)  # 等待执行
        pos = instrument.read_registers(0x0042, 2)
        print(f"Gripper opened to {pos[0]:.3f} m")
        
        # 设定夹爪关闭（最小位置 0mm）
        instrument.write_registers(0x0002, [0x0000, 0x0000])
        time.sleep(1)  # 等待执行
        pos = instrument.read_registers(0x0042, 2)
        print(f"Gripper closed to {pos[0]:.3f} m")
        
        # 读取夹爪状态
        status = instrument.read_register(0x0041, 0)
        states = {0: "At position", 1: "Moving", 2: "Holding object", 3: "Dropped object"}
        print(f"Gripper state: {states.get(status, 'Unknown')}" )
        
        # 🔥 设置夹爪目标位置：25mm
        position = 50  # 25mm
        ieee754_bytes = struct.pack('>f', position)  # 转换成 IEEE 754 4 字节
        high_word, low_word = struct.unpack('>HH', ieee754_bytes)  # 拆分高16位和低16位

        # 发送目标位置
        instrument.write_registers(0x0002, [high_word, low_word])
        time.sleep(1)  # 等待执行
        pos_data = instrument.read_registers(0x0042, 2)
        pos_bytes = struct.pack('>HH', pos_data[0], pos_data[1])
        position = struct.unpack('>f', pos_bytes)[0]
        print(f"Gripper moved to {position:.3f} mm")

    except Exception as e:
        print(f"Error during gripper test: {e}")

if __name__ == '__main__':
    test_gripper()
