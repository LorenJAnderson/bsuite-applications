from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
import numpy as np


scores = []
for i in range(1, 11):
    file = glob.glob("tensorboard/ent_0_00/A2C_" + str(i) + "/*")[0]
    rews = []
    for event in summary_iterator(file):
        for value in event.summary.value:
            if value.tag == "rollout/ep_rew_mean":
                rews.append(value.simple_value)
    rews.sort(reverse=True)
    scores.append(np.mean(rews[int(len(rews) * .00): int(len(rews) * .01)]))
print(np.mean(scores))


scores = []
for i in range(1, 11):
    file = glob.glob("tensorboard/ent_0_01/A2C_" + str(i) + "/*")[0]
    rews = []
    for event in summary_iterator(file):
        for value in event.summary.value:
            if value.tag == "rollout/ep_rew_mean":
                rews.append(value.simple_value)
    rews.sort(reverse=True)
    scores.append(np.mean(rews[int(len(rews) * .00): int(len(rews) * .01)]))
print(np.mean(scores))