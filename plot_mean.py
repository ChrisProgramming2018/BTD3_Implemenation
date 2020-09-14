from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt
import numpy as np


path = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3_seed_1_agent_TD3/events.out.tfevents.1599507860.architect.6470.0"
path1 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3_seed_2_agent_TD3/events.out.tfevents.1599507860.architect.6471.0"
path2 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3_seed_3_agent_TD3/events.out.tfevents.1599507860.architect.6472.0"
path3 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3_seed_4_agent_TD3/events.out.tfevents.1599507860.architect.6473.0"

p1="/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3lr_critic_0.001_update_freq_50_num_q_target_6_seed_1_agent_TD3_ad/events.out.tfevents.1599731008.architect.25365.0"
p2="/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3lr_critic_0.05lr_actor_0.005_update_freq_50_num_q_target_6_seed_2_agent_TD3_ad/events.out.tfevents.1599735305.architect.29334.0"
p3 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3lr_critic_0.05lr_actor_0.005_update_freq_50_num_q_target_6_seed_3_agent_TD3_ad/events.out.tfevents.1599735305.architect.29335.0"
p4 = "/home/architect/hiwi/bachelorProject/mujoco_env/vector/halfcheetah/runs/HalfCheetah-v3lr_critic_0.05lr_actor_0.005_update_freq_50_num_q_target_6_seed_4_agent_TD3_ad/events.out.tfevents.1599735305.architect.29336.0"

def createMean(path, path1, path2, path3, name="Reward"):
    value = []
    steps = []
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag == name:
                value.append(v.simple_value)
                steps.append(e.step)
    value1 = []
    steps1 = []
    for e in summary_iterator(path1):
        for v in e.summary.value:
            if v.tag == name:
                value1.append(v.simple_value)
                steps1.append(e.step)
    value2 = []
    steps2 = []
    for e in summary_iterator(path2):
        for v in e.summary.value:
            if v.tag == name:
                value2.append(v.simple_value)
                steps2.append(e.step)
    value3 = []
    steps3 = []
    for e in summary_iterator(path3):

        for v in e.summary.value:
            if v.tag == name:
                value3.append(v.simple_value)
                steps3.append(e.step)
    mean_value = []
    for v1, v2, v3, v4 in zip(value, value1, value2, value3):
        mean_value.append((v1+v2+v3+v4)/4.)
    var = []
    for v1, v2, v3, v4, mean in zip(value, value1, value2, value3, mean_value):
        summe = ((v1 -mean)**2 + (v2 -mean)**2 + (v3 -mean)**2 + (v4 -mean)**2)/4.
        var.append(np.sqrt(summe))
    max_mean = []
    min_mean = []
    for v, m in zip(var, mean_value):
        max_mean.append(m+v)
        min_mean.append(m-v)
    return mean_value, min_mean, max_mean



mean_value, min_mean, max_mean =createMean(path,path1,path2,path3)
#mean_value1, min_mean1, max_mean1 =createMean(p1,p2,p3,p4)

plt.plot(steps, min_mean, color='b')
plt.plot(steps, max_mean, color='b')
plt.fill_between(steps, min_mean, max_mean)
plt.plot(steps, mean_value,color='r')
plt.savefig("TD3vsTD3_ad.png")
