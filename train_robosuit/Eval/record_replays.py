"""
Render an eval's replayed episodes to one mp4 — in a process of its own.
========================================================================
The eval scripts collect the replayed trajectories and hand them here as an
.npz; nothing in this file knows about policies or tasks.

Why a separate process: the eval already holds an OpenGL context — the passive
viewer's, on its own thread. An offscreen context opened alongside it shares
libmujoco's GL function table, and on a machine with two GPUs (an AMD iGPU
driving the display, an NVIDIA dGPU) the two can land on different drivers.
Whichever initialises first wins, so from run to run the textured geoms — the
table, the walls — render black while untextured ones look fine. A process
with a single context cannot mix them.

Usage (the eval scripts call this; it is not meant to be run by hand):
    python record_replays.py replays.npz out.mp4
"""
import os
import sys

# Offscreen, no window — MUST precede `import mujoco`. Anything already set in
# the shell wins.
os.environ.setdefault("MUJOCO_GL", "egl")

import shutil
import subprocess

import numpy as np
import mujoco


def make_camera(model, p):
    """9 floats from the eval → MjvCamera; NaN = the viewer's opening view."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    if not np.isnan(p).any():
        cam.type, cam.fixedcamid, cam.trackbodyid = int(p[0]), int(p[1]), int(p[2])
        cam.lookat[:] = p[3:6]
        cam.distance, cam.azimuth, cam.elevation = p[6], p[7], p[8]
    return cam


def render(npz_path, out_path, width=640, height=480, border=6, hold=1.0):
    """One frame per control step at real-time fps; each episode framed
    green/red by its terminal-segment criterion and ending on a hold."""
    z = np.load(npz_path)
    model = mujoco.MjModel.from_xml_path(str(z['xml']))
    if 'marker' in z:                         # the goal site, made visible
        sid = int(z['marker'][0])
        model.site_size[sid, 0] = z['marker'][1]
        model.site_rgba[sid] = z['marker'][2:6]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, int(z['key_id']))
    mujoco.mj_forward(model, data)

    # The XML's offscreen buffer bounds the frame; libx264 wants even sizes.
    g = model.vis.global_
    w, h = min(width, g.offwidth) & ~1, min(height, g.offheight) & ~1
    renderer = mujoco.Renderer(model, h, w)
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = False                  # hide collision geoms
    opt.geomgroup[1] = True                   # show visual geoms

    fps = float(z['fps'])
    proc = subprocess.Popen(
        [shutil.which('ffmpeg'), '-y', '-loglevel', 'error',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
         '-r', f'{fps:g}', '-i', '-', '-an',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path],
        stdin=subprocess.PIPE)

    n_done = 0
    try:
        for qpos, mocap, ok, p in zip(z['qpos'], z['mocap'], z['ok'], z['cam']):
            if model.nmocap:
                data.mocap_pos[:] = mocap
            cam = make_camera(model, p)
            color = (40, 200, 40) if ok else (220, 40, 40)
            for q in qpos:
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam, scene_option=opt)
                f = renderer.render()
                for s in (np.s_[:border], np.s_[-border:],
                          np.s_[:, :border], np.s_[:, -border:]):
                    f[s] = color
                proc.stdin.write(f.tobytes())
            for _ in range(int(round(hold * fps))):
                proc.stdin.write(f.tobytes())
            n_done += 1
        proc.stdin.close()
    except BrokenPipeError:
        print(f"! ffmpeg stopped accepting frames for {out_path}")
    rc = proc.wait()
    renderer.close()

    if rc == 0:
        print(f"    ✓ recorded {n_done} episodes → {out_path}")
    else:
        print(f"! ffmpeg exited with {rc}; {out_path} may be incomplete")
    return rc


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(render(sys.argv[1], sys.argv[2]))
