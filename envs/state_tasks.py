"""
State variants of every robosuite task.
=======================================
Each class is `StateWarpEnv` mixed in front of the task's existing environment:

    class StateCubeStackingEnv(StateWarpEnv, CubeStackingEnv): ...

The method resolution order is

    StateCubeStackingEnv → StateWarpEnv → CubeStackingEnv → VisionWarpEnv

so StateWarpEnv wins for everything about the OBSERVATION and the ROLLOUT
(`build_obs`, `discover_indices`' slot ids, `_make_rollout`, `build`,
`_bundle`), while the task class keeps everything about the TASK — reward
composition, state randomisation, metrics and the success criterion.  None of
those ever touched pixels, so there is nothing to reimplement and nothing that
can drift between the two branches.

The bodies are empty on purpose.  What each task feeds the policy is declared in
`state_slots:` in configs/tasks_state.yaml, not here, so a new task needs a
config entry and (only if its physics logic is genuinely new) a class.
"""
from envs.state_base import StateWarpEnv

from envs.reach import ReachEnv
from envs.cube_stacking import CubeStackingEnv
from envs.pick_place import PickPlaceEnv
from envs.push_cube import PushCubeEnv
from envs.container import ContainerEnv
from envs.peg_insertion import PegInsertionEnv
from envs.articulated import ArticulatedEnv


class StateReachEnv(StateWarpEnv, ReachEnv):
    """Free-space reach. The target was already fed in as coordinates in the
    vision pipeline (too small a marker to localise on a 4x4 patch grid), so
    this is the one task the two branches observe identically."""


class StateCubeStackingEnv(StateWarpEnv, CubeStackingEnv):
    """Stack one cube on another, from both cubes' exact positions."""


class StatePickPlaceEnv(StateWarpEnv, PickPlaceEnv):
    """Carry a cube to a marker. Vision found the cube and was told the marker;
    here both are exact."""


class StatePushCubeEnv(StateWarpEnv, PushCubeEnv):
    """Push a cube to a goal. The standoff point the reach aims at is a function
    of the cube and goal positions, both of which are now given exactly."""


class StateContainerEnv(StateWarpEnv, ContainerEnv):
    """Three objects into a container. Slot order matches `per_object.objects`,
    so slot i is the object the i-th group of five segments works on."""


class StatePegInsertionEnv(StateWarpEnv, PegInsertionEnv):
    """Seat a peg in a slot. peg_grip and peg_bottom are two points on the same
    rigid body, so their difference gives the peg's tilt — which is what align
    and insert need, and why positions alone suffice."""


class StateArticulatedEnv(StateWarpEnv, ArticulatedEnv):
    """Drawer / window / dial / door. Two slots: the handle's exact position,
    and [joint, target, remaining travel]."""


#: `env:` value in configs/tasks_state.yaml → the class that implements it.
#: The keys match the vision registry's, so a task definition can be copied
#: between the two configs without renaming anything.
STATE_ENV_CLASSES = {
    'reach':          StateReachEnv,
    'cube_stacking':  StateCubeStackingEnv,
    'pick_place':     StatePickPlaceEnv,
    'push_cube':      StatePushCubeEnv,
    'container':      StateContainerEnv,
    'peg_insertion':  StatePegInsertionEnv,
    'articulated':    StateArticulatedEnv,
}
