import bsuite
from bsuite import sweep
import gym
from bsuite.utils import gym_wrapper

print(sweep.SWEEP)
print(sweep.BANDIT_NOISE)

for bsuite_id in sweep.MEMORY_LEN:
  env = bsuite.load_from_id(bsuite_id)
  print('bsuite_id={}, settings={}, num_episodes={}'
        .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

