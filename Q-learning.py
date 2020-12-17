import gym
import numpy as np

return_per_episode = []
env = gym.make('gym_cliffwalking:cliffwalking-v0')
env.reset()

board = np.zeros((4,12))
alpha = 0.9
gamma = 0.9
epsilon = 0.1
Q_values = np.ones((4*12,4))*.1
Q_values[1:11,:] *= -100
Q_values[6,:] = .1

def drawer(observation,board):
    board *= 0
    y = observation % 12
    x = (observation - y)/12
    if(x == 0):
        x = 3
    elif(x == 1):
        x = 2
    elif(x == 2):
        x = 1
    else:
        x = 0
    board[x,y] = 1
    print(board)
total_return = 0
observation = 0
for iteration in range(10000):
    #env.render()
    current_state = observation
    decision = np.random.choice(2, 1, p=[epsilon, 1 - epsilon])

    if (decision == 0):
        action = env.action_space.sample()
        #print(iteration)
    else:
        actions = np.argwhere(Q_values[current_state] == np.amax(Q_values[current_state]))
        action = np.random.randint(actions.shape[0])
        #print(actions)
        #print(action)
        action = actions[action][0]
        #print(action)

    observation, reward, done, info = env.step(action) # take a random action
    #drawer(observation,board)

    total_return += reward
    #print(observation)
    #Update Q-value
    if done:
        # print(reward)
        print(total_return)
        Q_values[current_state, action] += alpha * (reward - Q_values[current_state, action])
        env.reset()
        return_per_episode.append(total_return)
        total_return = 0
        #break
    else:
        Q_values[current_state,action] += alpha*(reward + gamma * np.max(Q_values[observation,:])-Q_values[current_state,action])
env.close()
print(Q_values)
graph = np.max(Q_values,axis=1)
decisions = np.argmax(Q_values,axis=1)
import matplotlib.pyplot as plt
plt.imshow(graph.reshape((4,12)))
plt.title('Q_values for Q-learning')
plt.show()
plt.plot(return_per_episode)
plt.title('Q-learning')
plt.xlabel('Episode')
plt.ylabel('Return per episode')
plt.show()
print(decisions.reshape((4,12)))