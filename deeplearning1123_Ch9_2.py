import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyperparameters
BATCH_SIZE = 32                                 
LR = 0.01                                       # learning rate
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount(衰減率)
TARGET_REPLACE_ITER = 100                       # How often the target network is updated
MEMORY_CAPACITY = 2000                          # Memory capacity
env = gym.make('CartPole-v1', render_mode="human").unwrapped
# env = gym.make('CartPole-v1').unwrapped         # Use the environment in the gym library: CartPole and unwrap the package
N_ACTIONS = env.action_space.n                  # Number of car movements (2)
N_STATES = env.observation_space.shape[0]       # Number of poles in state (4)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.out = nn.Linear(50, N_ACTIONS) 
    def forward(self, x):                                                       # Define the forward function (x is the state)
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value                                                    # Returns the action value

# Defining DQN Classes (Defining Two Networks)
class DQN(object):
    def __init__(self): 
        self.eval_net, self.target_net = Net(), Net()                           # Create two neural networks with Net: the evaluation network and the target network
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # Initialize the memory        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):                                                 # Defining the Action Selection Function (x is the state)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # Convert x to 32-bit floating point and add 1 dimension at dim=0
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)                            # By entering the state x, the action value is obtained by forward propagation
            action = torch.max(actions_value, 1)[1].data.numpy()                # Outputs the index of the maximum value of each row
            action = action[0]                                                  # The first number of output actions
        else:                                                                   
            action = np.random.randint(0, N_ACTIONS)                            # Here action is randomly equal to 0 or 1 (N_ACTIONS = 2)
        return action

    def store_transition(self, s, a, r, s_):                                    # Define a memory store function (input is a transition)
        transition = np.hstack((s, [a, r], s_))                                 # Splice arrays horizontally
        # If the memory bank is full, the old data is overwritten        
        index = self.memory_counter % MEMORY_CAPACITY                           # Gets the number of lines to be placed in the transition
        self.memory[index, :] = transition                                      # store the transition
        self.memory_counter += 1

    def learn(self):   # Defining Learning Functions (Start Learning When the Memory is Full)   
        # The target network parameters are updated
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # Triggered at the beginning, then every 100 steps           
            self.target_net.load_state_dict(self.eval_net.state_dict())         # Assign the parameters of the evaluation network to the target network
        self.learn_step_counter += 1
        # Extract batch data from the memory       
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 32 numbers are randomly drawn within [0, 2000) and may be repeated
        b_memory = self.memory[sample_index, :]                                 # Extract 32 transitions corresponding to 32 indexes and deposit them in b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # Convert to 32-bit floating point form and store it in the b_s, the shape of b_s is 32 rows and 4 columns
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        # Extract 32 a's, convert them to 64-bit integer (signed) form, and store them in the b_a (the reason why it is the LongTensor type is to facilitate the use of torch.gather later), b_a is 32 rows and 1 column
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        # The 32 R's are extracted into 32-bit floating point form and stored in the b_s, b_r is 32 rows and 1 column        
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # The 32 s_ were extracted, converted into 32-bit floating point form, and stored in the b_s, and the b_s_ was 32 rows and 4 columns
        # The evaluation and target values of 32 transitions were obtained, and the evaluation network parameters were updated using the loss function and optimizer
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s) By evaluating the network output, 32 rows of each b_s are output for a series of action values, and then .gather(1, b_a) represents the Q value extraction of the corresponding index b_a for each row for aggregation
        q_next = self.target_net(b_s_).detach()
        # q_next does not backpass errors, so detach;q_next represents a series of action values corresponding to each b_s_ of 32 rows output through the target network        
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0] means that only the maximum value of each row is returned, and no index (a one-dimensional tensor with length 32) is returned; view() means to turn the previously obtained one-dimensional tensor into the shape of (BATCH_SIZE, 1), and finally obtain the target value by the formula
        loss = self.loss_func(q_eval, q_target)
        # Enter 32 evaluation values and 32 target values, using the mean square loss function        
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()

dqn = DQN()

for i in range(400):
    print('<<<<<<<<<Episode: %s' % i)
    s = env.reset()[0]                                                     # Reset the environment
    episode_reward_sum = 0                                      # The total reward for the episode corresponding to the cycle is initialized

    while True:                                                         # Start an episode (Each cycle represents a step)        
        env.render()                                                    # Displays the experiment animation
        a = dqn.choose_action(s)
        s_, r, done, info, _ = env.step(a)                  # Perform actions and get feedback
        # Modify the reward to get the trained achievement faster
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # The closer the car is to the middle, the better
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5  # The more positive the pillars(柱子), the better
        new_r = r1 + r2

        dqn.store_transition(s, a, new_r, s_)                 # Store samples in replay memory
        episode_reward_sum += new_r                           # Step by step to add the reward for each step in an episode

        s = s_                                                # Update the status

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            break 
