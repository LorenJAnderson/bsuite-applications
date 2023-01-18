import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn
from bsuite.baselines.tf import boot_dqn

SAVE_PATH_DQN = 'reports/bsuite/full'
env = bsuite.load_and_record("bandit/0", save_path=SAVE_PATH_DQN, overwrite=True)
agent = dqn.default_agent(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec()
)
experiment.run(agent, env, num_episodes=env.bsuite_num_episodes)