"""
MJX Franka + Cube Simulation Test — with Live Viewer

Loads the Panda + single cube scene from MuJoCo Playground,
runs MJX simulation, and either:
  --live    : shows real-time interactive viewer (mouse ile döndür, zoom yap)
  (default) : renders to video file

Live viewer controls:
  - Sol tık + sürükle  : kamerayı döndür
  - Sağ tık + sürükle  : kamerayı kaydır (pan)
  - Scroll             : zoom in/out
  - Çift tık           : nesneye odaklan
  - SPACE              : pause/resume
  - ESC                : kapat
  - TAB                : kamera değiştir

Usage:
    source ~/mjx_diffrl_env/bin/activate

    # Canlı viewer (interaktif pencere):
    python mjx_franka_cube_live.py --live

    # Video kaydet (eski davranış):
    python mjx_franka_cube_live.py

    # Daha uzun simülasyon:
    python mjx_franka_cube_live.py --live --duration 10

    # macOS kullanıyorsan:
    mjpython mjx_franka_cube_live.py --live
"""

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MJX Franka + Cube Test")
parser.add_argument("--live", action="store_true",
                    help="Canlı interaktif viewer aç (video yerine)")
parser.add_argument("--duration", type=float, default=3.0,
                    help="Simülasyon süresi (saniye)")
parser.add_argument("--fps", type=int, default=30,
                    help="Render FPS")
parser.add_argument("--no-diff", action="store_true",
                    help="Differentiability testini atla")
parser.add_argument("--xml", type=str,
                    default="/home/emre/mjx_diffsim/franka_emika_panda/"
                            "mjx_single_cube.xml",
                    help="MJCF scene XML path")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCENE_XML = args.xml
SIM_DURATION = args.duration
RENDER_FPS = args.fps
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
VIDEO_PATH = "mjx_franka_cube_test.mp4"
LIVE_MODE = args.live

# ---------------------------------------------------------------------------
# Load MuJoCo model
# ---------------------------------------------------------------------------
print("Loading MuJoCo model...")
mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
mj_data = mujoco.MjData(mj_model)

# Load "home" keyframe
key_id = mj_model.keyframe("home").id
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)

print(f"Model loaded:")
print(f"  nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
print(f"  nbody={mj_model.nbody}")
print(f"  timestep={mj_model.opt.timestep}")
print(f"  bodies: {[mj_model.body(i).name for i in range(mj_model.nbody)]}")

# Print joint info
print(f"\nJoints ({mj_model.njnt}):")
for i in range(mj_model.njnt):
    jnt = mj_model.jnt(i)
    print(f"  [{i}] {jnt.name} type={jnt.type[0]} qposadr={jnt.qposadr[0]}")

# ---------------------------------------------------------------------------
# Put model on GPU via MJX
# ---------------------------------------------------------------------------
print("\nPutting model on GPU (MJX)...")
t0 = time.time()
mjx_model = mjx.put_model(mj_model)
print(f"  mjx.put_model: {time.time()-t0:.2f}s")

# Initialize data from keyframe
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mjx_data = mjx.put_data(mj_model, mj_data)
print(f"  mjx.put_data done")

# ---------------------------------------------------------------------------
# JIT compile step function
# ---------------------------------------------------------------------------
print("\nJIT compiling mjx.step...")
t0 = time.time()
jit_step = jax.jit(mjx.step)

# Warm-up compilation
mjx_data_next = jit_step(mjx_model, mjx_data)
mjx_data_next.qpos.block_until_ready()
print(f"  JIT compile: {time.time()-t0:.2f}s")

# ---------------------------------------------------------------------------
# Helper: MJX state → CPU MjData (viewer'ın anlayacağı format)
# ---------------------------------------------------------------------------
def mjx_state_to_mjdata(state, mj_model, mj_data_target):
    """MJX GPU state'ini CPU tarafındaki MjData'ya kopyalar.
    
    MJX simülasyonu JAX array'leri ile GPU'da çalışır.
    MuJoCo viewer ise CPU'daki MjData ile çalışır.
    Bu fonksiyon aradaki köprü — her frame'de çağrılır.
    """
    mj_data_target.qpos[:] = np.array(state.qpos)
    mj_data_target.qvel[:] = np.array(state.qvel)
    mujoco.mj_forward(mj_model, mj_data_target)

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
n_steps = int(SIM_DURATION / mj_model.opt.timestep)
print(f"\nRunning {n_steps} steps ({SIM_DURATION}s at dt={mj_model.opt.timestep})...")

# Reset
mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
mjx_data = mjx.put_data(mj_model, mj_data)

# Simple control: hold home position
home_ctrl = jnp.array(mj_data.ctrl)

# How many physics steps per render frame
render_every = max(1, int(1.0 / (RENDER_FPS * mj_model.opt.timestep)))

if LIVE_MODE:
    # ===================================================================
    # CANLI VIEWER MODU
    # ===================================================================
    # launch_passive: pencereyi açar ama scripti bloklamaz.
    # Sen mj_data'yı güncelle → viewer.sync() çağır → pencere güncellenir.
    # ===================================================================
    from mujoco import viewer as mj_viewer
    
    print("\n" + "=" * 60)
    print("CANLI VIEWER AÇILIYOR")
    print("  Sol tık + sürükle  → kamerayı döndür")
    print("  Sağ tık + sürükle  → kamerayı kaydır")
    print("  Scroll             → zoom")
    print("  SPACE              → pause/resume")
    print("  ESC                → kapat")
    print("=" * 60 + "\n")
    
    # CPU tarafında render için kullanacağımız MjData
    mj_data_render = mujoco.MjData(mj_model)
    
    # İlk state'i kopyala (viewer açılınca robot görünsün)
    mjx_state_to_mjdata(mjx_data, mj_model, mj_data_render)
    
    # Viewer'ı aç — mj_data_render üzerinden çalışacak
    viewer_handle = mj_viewer.launch_passive(mj_model, mj_data_render)
    
    t0 = time.time()
    state = mjx_data
    
    for i in range(n_steps):
        # Viewer kapatıldı mı kontrol et
        if not viewer_handle.is_running():
            print("\nViewer kapatıldı, simülasyon durduruluyor.")
            break
        
        # MJX physics step (GPU'da)
        state = state.replace(ctrl=home_ctrl)
        state = jit_step(mjx_model, state)
        
        # Render frame'i geldi mi?
        if i % render_every == 0:
            # GPU → CPU: state'i MjData'ya kopyala
            mjx_state_to_mjdata(state, mj_model, mj_data_render)
            
            # Viewer'a "güncellendi" de
            viewer_handle.sync()
            
            # Gerçek zamanlı hız için bekle
            # (Physics çok hızlı çalışır, yoksa video 100x hızlı oynar)
            elapsed = time.time() - t0
            sim_clock = (i + 1) * mj_model.opt.timestep
            sleep_time = sim_clock - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Progress
        if n_steps > 10 and i % (n_steps // 10) == 0:
            qpos = np.array(state.qpos)
            cube_pos = qpos[9:12]
            print(f"  step {i:5d}/{n_steps}: cube=[{cube_pos[0]:.3f}, "
                  f"{cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    
    state_qpos = state.qpos if viewer_handle.is_running() else state.qpos
    state_qpos_np = np.array(state_qpos)
    
    sim_time = time.time() - t0
    print(f"\nSimulation done: {sim_time:.2f}s ({i/max(sim_time,0.001):.0f} steps/s)")
    
    # Viewer açık kalsın — kullanıcı ESC ile kapatsın
    if viewer_handle.is_running():
        print("\nSimülasyon bitti. Viewer hâlâ açık — ESC ile kapat.")
        print("(Son frame'de donmuş halde duruyor, kamerayı döndürebilirsin)")
        try:
            while viewer_handle.is_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nCtrl+C — çıkılıyor.")
    
    # Final state
    cube_pos_final = state_qpos_np[9:12]
    print(f"Final cube pos: [{cube_pos_final[0]:.4f}, {cube_pos_final[1]:.4f}, "
          f"{cube_pos_final[2]:.4f}]")

else:
    # ===================================================================
    # VIDEO MODU (eski davranış)
    # ===================================================================
    import mediapy as media
    
    render_states = []
    
    t0 = time.time()
    state = mjx_data
    for i in range(n_steps):
        state = state.replace(ctrl=home_ctrl)
        state = jit_step(mjx_model, state)
        
        if i % render_every == 0:
            render_states.append(state)
        
        if n_steps > 10 and i % (n_steps // 10) == 0:
            qpos = np.array(state.qpos)
            cube_pos = qpos[9:12]
            print(f"  step {i:5d}/{n_steps}: cube=[{cube_pos[0]:.3f}, "
                  f"{cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    
    state.qpos.block_until_ready()
    sim_time = time.time() - t0
    print(f"\nSimulation done: {sim_time:.2f}s ({n_steps/sim_time:.0f} steps/s)")
    
    # Final state
    qpos_final = np.array(state.qpos)
    cube_pos_final = qpos_final[9:12]
    print(f"Final cube pos: [{cube_pos_final[0]:.4f}, {cube_pos_final[1]:.4f}, "
          f"{cube_pos_final[2]:.4f}]")
    print(f"Cube Z stable (near 0.03): {abs(cube_pos_final[2] - 0.03) < 0.01}")
    
    # Render video
    print(f"\nRendering {len(render_states)} frames...")
    renderer = mujoco.Renderer(mj_model, width=RENDER_WIDTH, height=RENDER_HEIGHT)
    
    frames = []
    for state in render_states:
        mj_data_render = mujoco.MjData(mj_model)
        mjx_state_to_mjdata(state, mj_model, mj_data_render)
        renderer.update_scene(mj_data_render)
        frames.append(renderer.render())
    
    renderer.close()
    
    media.write_video(VIDEO_PATH, frames, fps=RENDER_FPS)
    print(f"Video saved: {VIDEO_PATH}")

# ---------------------------------------------------------------------------
# Test: differentiability (optional)
# ---------------------------------------------------------------------------
if not args.no_diff:
    print("\n--- Differentiability Test ---")
    
    mj_model_diff = mujoco.MjModel.from_xml_path(SCENE_XML)
    mj_model_diff.opt.iterations = 1
    mj_model_diff.opt.ls_iterations = 4
    
    mujoco.mj_resetDataKeyframe(mj_model_diff, mj_data, key_id)
    mjx_model_diff = mjx.put_model(mj_model_diff)
    mjx_data_diff = mjx.put_data(mj_model_diff, mj_data)
    
    def loss_fn(ctrl):
        """Simple test: does jax.grad work through mjx.step?"""
        state = mjx_data_diff.replace(ctrl=ctrl)
        for _ in range(10):
            state = mjx.step(mjx_model_diff, state)
        cube_z = state.qpos[11]
        return -cube_z
    
    print("Computing gradient through 10 MJX steps...")
    t0 = time.time()
    grad_fn = jax.jit(jax.grad(loss_fn))
    grad = grad_fn(home_ctrl)
    grad.block_until_ready()
    print(f"  jax.grad + JIT: {time.time()-t0:.2f}s")
    print(f"  Gradient: {np.array(grad)}")
    print(f"  Grad norm: {np.linalg.norm(np.array(grad)):.6f}")
    print(f"  Non-zero: {np.count_nonzero(np.abs(np.array(grad)) > 1e-10)}"
          f"/{len(np.array(grad))}")
    
    if np.linalg.norm(np.array(grad)) > 1e-10:
        print("\n✓ MJX is DIFFERENTIABLE! Gradients flow through physics.")
    else:
        print("\n✗ Gradients are zero — something is wrong.")
    
    print("\n--- Speed Test (after JIT) ---")
    t0 = time.time()
    for i in range(10):
        grad = grad_fn(home_ctrl)
        grad.block_until_ready()
    elapsed = time.time() - t0
    print(f"  10 grad calls: {elapsed:.2f}s ({elapsed/10:.3f}s per call)")