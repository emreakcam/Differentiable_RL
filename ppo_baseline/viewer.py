"""
Passive MuJoCo viewer for the PPO baseline.
===========================================
The same idea as the DiffRL trainers' viewer, adapted to this rollout: there an
episode arrives as (chunk, step, substep, nq); here `step_env_jit` returns one
state per control step, so a trajectory is (step, nq) and replays at frame_dt.

Loads its own MjModel from the XML rather than reusing the task's — the task's
copy has had non-visual geoms pushed to group 3, which does not suit a human.
"""
import time

import mujoco


class TrajectoryViewer:
    """Replays one env's qpos trajectory in roughly real time."""

    def __init__(self, task, frame_dt):
        from mujoco import viewer as mj_viewer
        self.model = mujoco.MjModel.from_xml_path(task['xml'])
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, task['key_id'])
        mujoco.mj_forward(self.model, self.data)
        self.handle = mj_viewer.launch_passive(self.model, self.data)
        self.handle.opt.geomgroup[0] = False      # hide collision geoms
        self.handle.opt.geomgroup[1] = True       # show visual geoms
        self.handle.sync()
        self.frame_dt = float(frame_dt)

    @property
    def running(self):
        return self.handle.is_running()

    def replay(self, qpos, skip=1):
        """qpos: (steps, nq) for a single environment."""
        if not self.running:
            return
        for i in range(0, qpos.shape[0], skip):
            if not self.running:
                return
            self.data.qpos[:] = qpos[i]
            mujoco.mj_forward(self.model, self.data)
            self.handle.sync()
            time.sleep(self.frame_dt * skip)

    def close(self):
        try:
            self.handle.close()
        except Exception:
            pass
