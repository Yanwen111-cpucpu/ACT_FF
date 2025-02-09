import time
import threading
import rospy
from cam_pub import RealSenseCamera
from fanuc_controller import FanucPub
from gripper_controller import GripperController

class ROSLauncher:
    """ 直接启动相机、机械臂和 Gripper 的 Python 类 """

    def __init__(self):
        rospy.init_node("robot_system", anonymous=True)  # 统一初始化 ROS 节点

        # 存储所有对象
        self.cameras = []
        self.fanuc_controller = None
        self.gripper_controller = None

    def start_cameras(self):
        """ 启动所有 RealSense 相机 """
        from pyrealsense2 import context
        ctx = context()
        serials = [dev.get_info(2) for dev in ctx.query_devices()]  # 获取所有相机序列号

        if not serials:
            rospy.logerr("❌ No Intel RealSense devices found!")
            return

        for serial in serials:
            cam = RealSenseCamera(serial)
            self.cameras.append(cam)
            time.sleep(1)  # 避免资源冲突

        rospy.loginfo(f"✅ 已启动 {len(self.cameras)} 个 RealSense 相机")

    def start_fanuc(self):
        """ 启动 Fanuc 机械臂控制器 """
        self.fanuc_controller = FanucPub()
        rospy.loginfo("✅ Fanuc 机械臂已启动")

    def start_gripper(self):
        """ 启动 Gripper 控制器 """
        self.gripper_controller = GripperController()
        rospy.loginfo("✅ Gripper 已启动")

    def stop_all(self):
        """ 停止所有进程 """
        rospy.loginfo("\n[INFO] 正在停止所有 ROS 进程...")

        # 停止相机
        for cam in self.cameras:
            cam.stop_camera()
        rospy.loginfo("[INFO] 所有相机已关闭")

        # 停止 Fanuc 机械臂
        if self.fanuc_controller:
            del self.fanuc_controller
        rospy.loginfo("[INFO] Fanuc 机械臂已关闭")

        # 停止 Gripper
        if self.gripper_controller:
            del self.gripper_controller  # 清理对象
        rospy.loginfo("[INFO] Gripper 已关闭")

def main():
    launcher = ROSLauncher()

    # 使用线程启动各个设备，避免阻塞
    cam_thread = threading.Thread(target=launcher.start_cameras)
    fanuc_thread = threading.Thread(target=launcher.start_fanuc)
    gripper_thread = threading.Thread(target=launcher.start_gripper)

    cam_thread.start()
    fanuc_thread.start()
    gripper_thread.start()

    cam_thread.join()
    fanuc_thread.join()
    gripper_thread.join()

    rospy.loginfo("✅ 所有系统已启动，按 Ctrl+C 退出")

    try:
        rospy.spin()  # 保持 ROS 运行
    except KeyboardInterrupt:
        launcher.stop_all()

if __name__ == "__main__":
    main()
