# NOTE: do not name a module here `models`, `envs` or `helpers` — this
# directory goes on sys.path ahead of the repo root and would shadow them.
"""
rsl_rl actor/critic wrapping the trunk + per-primitive-head network.
====================================================================
rsl_rl's `MLPModel` already provides everything PPO asks of a policy — the
Gaussian, log-probs, entropy, KL, serialization — and builds its network as one
swappable `self.mlp` submodule.  So the actor here is that class with its
generic MLP replaced by TrunkHeadNet, and nothing about PPO's view of the policy
changes.

The critic is left as rsl_rl's stock MLP.  A value function is something PPO
needs and the DiffRL arm has no counterpart to, so there is no "same model"
constraint to honour: giving it primitive-specific heads would be inventing an
architecture the comparison never asked about.
"""
import torch

from rsl_rl.models import MLPModel

from ppo_baseline.torch_policy import TrunkHeadNet


class TrunkHeadActor(MLPModel):
    """MLPModel whose network is the DiffRL policy's trunk + heads.

    `state_obs_dim` is the observation width WITHOUT the primitive one-hot that
    the wrapper appends: the trunk must see only the observation, exactly as the
    JAX trunk does, and the one-hot must reach the head selector untouched.
    TrunkHeadNet splits the latent on that boundary.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 state_obs_dim, n_prims, trunk_hidden, head_widths,
                 head_layers, finger_biases, n_joints=7,
                 distribution_cfg=None, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim,
                         hidden_dims=(64,),      # discarded below
                         activation="elu",
                         obs_normalization=False,
                         distribution_cfg=distribution_cfg)
        # GaussianDistribution.input_dim == output_dim (its std is a separate
        # learnable parameter), so the network emits exactly `output_dim`.
        out_dim = (self.distribution.input_dim if self.distribution is not None
                   else output_dim)
        self.mlp = TrunkHeadNet(
            obs_dim=state_obs_dim, n_prims=n_prims, trunk_hidden=trunk_hidden,
            head_widths=head_widths, head_layers=head_layers,
            finger_biases=finger_biases, act_dim=out_dim, n_joints=n_joints)

    def update_normalization(self, obs):
        """No-op: the wrapper normalises with the run's own StateObsRMS.

        rsl_rl's EmpiricalNormalization has no variance floor, and the padded
        object slots are constant zero — without the floor those dims would be
        scaled differently here than in the DiffRL arm, on exactly the
        coordinates that never move.
        """
        return
