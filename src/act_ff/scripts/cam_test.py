import pyrealsense2 as rs
import numpy as np
import cv2

def test_realsense():
    """ 测试 RealSense 相机是否正常工作 """
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置相机分辨率、帧率
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        # 启动相机
        pipeline.start(config)
        print("✅ RealSense Camera started successfully.")

        while True:
            # 获取帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # 转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 显示图像
            cv2.imshow("RealSense Camera", color_image)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense Camera Stopped.")

if __name__ == "__main__":
    test_realsense()
