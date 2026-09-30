import mujoco
from mujoco import viewer as mj_viewer
import numpy as np
import time

XML_PATH = "assets/common/franka_emika_panda/mjx_revolute_door.xml"

mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data = mujoco.MjData(mj_model)

key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mj_data.ctrl[:] = mj_model.keyframe("home").ctrl
mujoco.mj_forward(mj_model, mj_data)

hand_body_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
ee_site_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

try:
    handle_site_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "leftdoor_handle")
except:
    handle_site_idx = None

print("Use Control tab sliders to move joints.")
print("EE quaternion prints every 2 seconds. Close when done.\n")

handle = mj_viewer.launch_passive(mj_model, mj_data)

last_print = 0
while handle.is_running():
    # Map ctrl sliders directly to qpos — no physics, no falling
    mj_data.qpos[:7] = mj_data.ctrl[:7]
    mj_data.qpos[7] = mj_data.ctrl[7]
    mj_data.qpos[8] = mj_data.ctrl[7]
    mujoco.mj_forward(mj_model, mj_data)
    handle.sync()

    now = time.time()
    if now - last_print > 2.0:
        ee_quat = mj_data.xquat[hand_body_idx]
        ee_pos = mj_data.site_xpos[ee_site_idx]

        print(f"  ee_quat=[{ee_quat[0]:.4f}, {ee_quat[1]:.4f}, "
              f"{ee_quat[2]:.4f}, {ee_quat[3]:.4f}]  "
              f"ee_pos=[{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]",
              end="")
        if handle_site_idx is not None:
            hp = mj_data.site_xpos[handle_site_idx]
            print(f"  handle=[{hp[0]:.4f}, {hp[1]:.4f}, {hp[2]:.4f}]", end="")
        print()
        last_print = now

    time.sleep(0.02)

ee_quat = mj_data.xquat[hand_body_idx]
ee_pos = mj_data.site_xpos[ee_site_idx]
q = mj_data.qpos[:7]

print(f"\n{'='*60}")
print(f"  TARGET_QUAT = jnp.array([{ee_quat[0]:.4f}, {ee_quat[1]:.4f}, {ee_quat[2]:.4f}, {ee_quat[3]:.4f}])")
print(f"  ee_pos  = [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
print(f"  q       = [{', '.join(f'{v:.4f}' for v in q)}]")
print(f"{'='*60}")