# Report
### Implementation
A Deep Q-Learning (DQN) agent was created using PyTorch. The file  ```dqn_agent.py``` includes two classes: Agent and ReplayBuffer. The DQN agent is based on the agent described in [this paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). The replay buffer is the agent's memory, where it stores the transitions from a state with an action to the next state with its reward.

The file ```model.py``` consists of a QNetwork class which uses PyTorch to implement a deep neural network. The model consists of three linear modules which applies a linear function to the input. Between these three linear modules, there are two ReLU layers to speed up the training.

The file ```collect-bananas.ipynb``` consists of the runnable code that puts the project together. Firstly, the environment and agent is instantiated. Then, the agent is trained by letting the agent act inside the environment for up to 1000 episodes (until a mean cumulative reward of 15* for the last 100 episodes is reached). The weights of the trained agent are saved in a file named ```model_weigths.pth```. Lastly, the trained agent is tested for 100 additional episodes and the mean score of these episodes are reported.

*The reason why 15 is chosen instead of 13 (as described in the project goal) is because there is such a great variety to the score obtained after a episode so by setting a higher goal it's likelier that the agent passes the final test.

### Learning Algorithm
Every neural network consists of nodes (neurons) associated with weights. When training a neural network, one refers to the act of updating the weights. During training, a training data set is used and the output is then compared to an answer or evaluated in some other way. The weights are adjusted depending on how good/bad the output is or how close it is to an answer. 

In this project, a DQL algorithm is used to make the agent decide what step to take. Deep Q-learning algorithms are benificial compared to non-deep Q-learning algoritms when there are a great number of states and/or actions since it requires less memory/experience and time for training. The specific learning algorithm used for this project is the DQL algorithm that was introduced in [this paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). The algorithm can be seen in *Fig. 1* below. 

![DQL Algorithm](https://github.com/Johannathorsen/collect-bananas/blob/master/Media/DQL_algorithm.png)
*Fig. 1. The DQN algorithm in pseudocode, obtained from [this paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).*

For this task, the hyperparameters for the DQN agent is set to:
 - **BUFFER_SIZE = int(1e5**) - replay buffer size
 - **BATCH_SIZE = 64** - minibatch size
 - **GAMMA = 0.99** - discount factor
 - **TAU = 1e-3** - for soft update of target parameters
 - **LR = 5e-4** - learning rate
 - **UPDATE_EVERY = 4** - how often to update the network

The fixed parameters for this project is:
 - **agents = 1** - number of agents 
 - **action_size = 4** - the number of actions an agent can choose between
 - **state_size = 37** - the length of each state vector
 - **max_t = 300** - maximum number of timesteps per episode

The parameters used for training the agent in the environment:

 - **n_episodes = 1000** - maximum number of training episodes
 -  **eps_start = 1.0** - starting value of epsilon, for epsilon-greedy action selection
 -  **eps_end = 0.01** - minimum value of epsilon
 -  **eps_decay= 0.995** - multiplicative factor (per episode) for decreasing epsilon

The target training score was set to reach an average of +15 for 100 consecutive episodes. The goal was to reach an average of +13 but since the score can vary greatly from one episode to another the target was set a little bit higher increase the chance of getting an average score above +13 for the testing round.

### Result
A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.

In the final test, the agent was placed in the environment to perform 100 episodes. The average score during these test was +15.15.

![Final test scores](https://github.com/Johannathorsen/collect-bananas/blob/master/Media/scores_during_testing.png)
*Fig. 2. The scores together with their corresponding episode in the final test.*

To achieve this, the agent was trained for 815 episodes. When the agent stopped its training, the average score for the last 100 episodes were +15.02.

![Training scores](https://github.com/Johannathorsen/collect-bananas/blob/master/Media/scores_during_training.png)
*Fig. 3. The scores together with their corresponding episode during the training.*

To experience the improvement the agent did during the training, two gif:s are inserted below. The first one is a recording from when the agent moved around in the environment without training and the second one during testing (when all training was done).

![Untrained agent](https://github.com/Johannathorsen/collect-bananas/blob/master/Media/untrained_collector.gif)
*Fig. 4. The untrained agent moving around in the environment.*

![Trained agent](https://github.com/Johannathorsen/collect-bananas/blob/master/Media/trained_collector.gif)
*Fig. 5. The trained agent moving around in the environment.*

### Ideas for Future Work
As seen in *Fig 2*, the spread of the cumulative reward that the agent gathered in each episode is quite big. Since the agent was limited to only performing 300 actions in each episode, the cumulative reward is of course dependent on how the bananas were distributed in the space. But in the episodes with the lowest scores, it is often due to the agent being stuck in a loop, see *Fig 6* below.

![Agent is stuck](https://github.com/Johannathorsen/collect-bananas/blob/master/Media/trained_collector_stuck.gif)
*Fig 6. The agent is doing the same moves over and over again and not getting anywhere.*

This is because the agent, when its not in training, only takes a state as an input and gives an action as an output. In training, there is a chance that the agent will perform a random action and hence it can escape these loops. But in testing, the actions are never choosen by chance.

To reduce the risk for these loops, we should add some randomness to the testing too. Either in a similar way as for training (by utilizing epsilon) or by saving the states with their score and action for each time step. If the agent has experienced the same state earlier in the episode and it still has the same amount of scores, the agent should choose another random action among the actions that is hasn't done at the same state earlier.

The performance could also be improved by exploring modifications of the hyperparameters or by changing the DQN. Examples of changes could be:

 - implementing a double DQN (which reduces the overestimation of the action values)
 - implementing a prioritized experience replay (which can make the agent learn faster)
 - replace the DQN with a dueling DQN (which reduce unnecessary learning)