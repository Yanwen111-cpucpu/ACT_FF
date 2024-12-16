from dm_control import suite
from dm_control import viewer

# 加载 MuJoCo 自带的 cartpole 模型
env = suite.load(domain_name="cartpole", task_name="swingup")

# 测试仿真函数
def test_simulation(steps=100):
    print("Testing MuJoCo simulation with cartpole model...")
    time_step = env.reset()  # 重置环境
    try:
        for step in range(steps):
            action = env.action_spec().sample()  # 随机生成一个动作
            time_step = env.step(action)  # 执行一个仿真步骤

            # 打印观测结果
            print(f"Step {step + 1}/{steps}")
            print("Observations:", time_step.observation)

            # 渲染图像
            pixels = env.physics.render(height=480, width=640, camera_id=0)
            if pixels is not None:
                print("Rendered image shape:", pixels.shape)

    except Exception as e:
        print("Error during simulation:", e)
    else:
        print("Simulation test completed successfully.")

# 运行交互式查看器，观察仿真过程
viewer.launch(lambda: env)

# 执行仿真测试
test_simulation(steps=50)
