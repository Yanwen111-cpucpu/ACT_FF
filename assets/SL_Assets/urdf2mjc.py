from urdf2mjcf import run
import os

os.environ['TEMP'] = 'D:/temp'
os.environ['TMP'] = 'D:/temp'

run(
    urdf_path="Gripper_sl.urdf",
    mjcf_path="Gripper_sl.mjcf",
    copy_meshes=True,
)