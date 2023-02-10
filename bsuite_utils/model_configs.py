from typing import NamedTuple

from sb3_contrib import RecurrentPPO, QRDQN
from stable_baselines3 import DQN, A2C, PPO

from bsuite_utils.custom_models import BSuiteDQNShim, FrameSkipWrapper, FrameStackWrapper, NormalizeWrapper
from bsuite_utils.mnist_wrapper import MNISTWrapper, CustomCNNPolicy


class ModelConfig(NamedTuple):
    name: str
    cls: type
    policy: str = "MlpPolicy"
    env_wrapper: type = None
    kwargs: dict = dict()
    wrapper_kwargs: dict = dict()


_dqn_default_kwargs = dict(learning_starts=1000, learning_rate=7e-4, buffer_size=10_000)
_ppo_default_kwargs = dict(learning_rate=0.0007, ent_coef=0.01, n_steps=128)

# 1.1
dqn_default = [ModelConfig(name="DQN_default", cls=DQN, kwargs=_dqn_default_kwargs), ]
a2c_default = [ModelConfig(name="A2C_default", cls=A2C)]
ppo_default = [ModelConfig(name="PPO_default", cls=PPO, kwargs=_ppo_default_kwargs), ]
# 1.2
dqn_alternate_implementations = [
    ModelConfig(name="DQN_BSuite", cls=BSuiteDQNShim)
    # TODO can add rllib implementation here as well if time
]
# 1.3
dqn_alternate_buffsizes = [
    ModelConfig(f"DQN_buf{x}", cls=DQN, kwargs={**_dqn_default_kwargs, "buffer_size": x}) for x in [100, 1_000, 100_000]
]

# 2
# dqn_life = [ModelConfig(name="DQN_life", cls=DQN, kwargs=_dqn_default_kwargs, env_wrapper=LifeWrapper)]
dqn_frameskip_variants = [
    ModelConfig(name=f"DQN_frameskip{x}", cls=DQN, kwargs=_dqn_default_kwargs, env_wrapper=FrameSkipWrapper,
                wrapper_kwargs={"n_skip": x}) for x in [2, 4, 6]]
dqn_normalize = [ModelConfig(name="DQN_normalize", cls=DQN, kwargs=_dqn_default_kwargs, env_wrapper=NormalizeWrapper)]
dqn_framestack = [
    ModelConfig(name="DQN_framestack", cls=DQN, kwargs=_dqn_default_kwargs, env_wrapper=FrameStackWrapper)]

# 3
ppo_entropy_variants = [ModelConfig(name=f"PPO_ent{x}", cls=PPO, kwargs={**_ppo_default_kwargs, "ent_coef": x}) for x in
                        [0.0, 0.001, 0.1]]
dqn_lr_variants = [
    ModelConfig(name=f"DQN_lr{x}", cls=DQN, kwargs={**_dqn_default_kwargs, "learning_rate": x}) for x in
    [10 ** (-x) for x in [0, 1, 2, 4, 6]]
]
dqn_epsilon_variants = [
    ModelConfig(name=f"DQN_eps{x}", cls=DQN, kwargs={**_dqn_default_kwargs, "exploration_fraction": x}) for x in
    [0.0, 0.05, 0.2, 0.4, 0.6]
]

# 4
dqn_bad_gammas = [
    ModelConfig(name=f"DQN_gamma{x}", cls=DQN, kwargs={**_dqn_default_kwargs, **dict(gamma=x)}) for x in [0.5, 1.0, 2.0, 3.0, 5.0]
]
# dqn_bad_update_intervals = [
#     ModelConfig(name=f"DQN_update_interval{x}", cls=DQN, kwargs={**_dqn_default_kwargs, **dict(target_update_interval=x)}) for x in [1_000_000, 2_000_000]
# ]
ppo_bad_lr = [
    ModelConfig(name=f"PPO_lr{x}", cls=PPO, kwargs={**_ppo_default_kwargs, "learning_rate": x}) for x in [1e3]
]
dqn_bad_burnin = [
    ModelConfig(name=f"DQN_burn{x}", cls=DQN, kwargs={**_dqn_default_kwargs, "learning_starts": x}) for x
    in [10_000]
]

# 5
ppo_rnn = [ModelConfig(name="RecurrentPPO", cls=RecurrentPPO, policy="MlpLstmPolicy",
                       kwargs={**_ppo_default_kwargs, "batch_size": 64})]
dqn_qrdn = [ModelConfig(name="QRDQN", cls=QRDQN, kwargs=_dqn_default_kwargs)]
dqn_qrdn_matched_parameters = [ModelConfig(name="QRDQN_MATCH", cls=QRDQN, kwargs={**_dqn_default_kwargs, **dict(
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
)})]

dqn_cnn_variants = [
    ModelConfig(
        name=f"DQN_CNN_{x}", cls=DQN, policy=CustomCNNPolicy, kwargs={**_dqn_default_kwargs, "policy_kwargs": dict(scale=x)}, env_wrapper=MNISTWrapper,
    )
    for x in ['small', 'medium', 'large']
]
