import gym
from gym.envs.registration import register
import sys, tty, termios
import numpy as np
import random as pr
import matplotlib.pyplot as plt

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP =3

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT,
}

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery' : False}
)

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000
rList = []

#discount factor
dis = .99

for i in range(num_episodes):
    #decaying  E-greedy
    e = 1. / ((i // 100) +1)
    
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        #action = np.argmax(Q[state,:]+np.random.randn(1,env.action_space.n) / (i+1))
        #Q[state,action] = reward+dis*max(Q[new_state,:])
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        new_state, reward, done, _ = env.step(action)
        Q[state,action] = reward + dis*np.max(Q[new_state,:])
        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)),rList, color="blue")
plt.show()

#env.render()
"""
while True:
    key = inkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
    action = arrow_keys[key]
    state,reward,done,info = env.step(action)
    env.render()
    print("State: ",state, "Action: ", action, "Rewar:", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break

"""
