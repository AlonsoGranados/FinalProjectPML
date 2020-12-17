import gym
import numpy as np

return_per_episode = []
env = gym.make('gym_cliffwalking:cliffwalking-v0')
env.reset()

entropy = 1
board = np.zeros((4,12))
alpha = 0.9
gamma = 0.9
epsilon = 0.1
Q_values = np.ones((4*12,4))*.1
Q_values[1:11,:] *= -100
Q_values[6,:] = .1

entropy_state = np.ones(4*12)

def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def drawer(observation,board):
    #board *= 0
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
    board[x,y] += 1
    return board
total_return = 0
observation = 0
for iteration in range(20000):
    current_state = observation

    # soft_probabilities = softmax(Q_values[current_state]/entropy)
    # state
    soft_probabilities = softmax(Q_values[current_state] / entropy_state[current_state])

    if (iteration % 1000 == 999):
    #     entropy /= 2
        entropy_state[current_state] /= 2

    #print(soft_probabilities)
    action = np.random.choice(4, 1, p=soft_probabilities)

    observation, reward, done, info = env.step(action) # take a random action
    board = drawer(observation,board)
    # if (reward == -100):
    #     print(Q_values[current_state])

    total_return += reward
    #print(observation)
    #Update Q-value



    if done:
        if(reward == 10):
           print(reward)
        # print(board)
        Q_values[current_state, action] += alpha * (reward - Q_values[current_state, action])
        board *= 0
        print(total_return)
        return_per_episode.append(total_return)
        env.reset()
        total_return = 0
        #break
    else:
        soft = np.log(np.sum(np.exp(Q_values[observation]) * soft_probabilities))
        # soft = np.max((Q_values[observation]))
        Q_values[current_state, action] += alpha * (reward + gamma * soft - Q_values[current_state, action])


env.close()
print(Q_values)
graph = np.max(Q_values,axis=1)
decisions = np.argmax(Q_values,axis=1)
import matplotlib.pyplot as plt
plt.imshow(graph.reshape((4,12)))
plt.title('Q_values for State dependent temperature')
plt.show()
plt.plot(return_per_episode)
plt.title('State dependent temperature')
plt.xlabel('Episode')
plt.ylabel('Return per episode')
plt.show()
print(decisions.reshape((4,12)))
print(graph.reshape((4,12)))