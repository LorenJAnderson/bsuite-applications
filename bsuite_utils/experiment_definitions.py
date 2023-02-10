from typing import List

from bsuite_utils.model_configs import *


class ExperimentConfig(NamedTuple):
    id: str  # Must not contain path-hostile characters. Preferably no spaces.
    model_configs: List[ModelConfig]


EXPERIMENTS = [
    ExperimentConfig("1.1", dqn_default + a2c_default + ppo_default),
    ExperimentConfig("1.2", dqn_default + dqn_alternate_implementations),
    ExperimentConfig("1.3", dqn_default + dqn_alternate_buffsizes),

    ExperimentConfig("2.1", dqn_default + dqn_framestack),
    ExperimentConfig("2.2", dqn_default + dqn_normalize),
    ExperimentConfig("2.3", dqn_default + dqn_frameskip_variants),

    ExperimentConfig("3.1", ppo_default + ppo_entropy_variants),
    ExperimentConfig("3.2", dqn_default + dqn_lr_variants),
    ExperimentConfig("3.3", dqn_default + dqn_epsilon_variants),

    ExperimentConfig("4.1", dqn_default + dqn_bad_gamma),
    ExperimentConfig("4.2", ppo_default + ppo_bad_lr),
    ExperimentConfig("4.3", dqn_default + dqn_bad_burnin),

    ExperimentConfig("5.1", ppo_default + ppo_rnn),
    ExperimentConfig("5.2", dqn_default + dqn_qrdn + dqn_qrdn_matched_parameters),
    ExperimentConfig("5.3", dqn_default + dqn_cnn),
]

ID_EXPERIMENT_MAP = {exp.id: exp for exp in EXPERIMENTS}
