import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network import Qnet
from buffer import ReplayBuffer
import utils

#### state = {"dealer" : {"ace" : int, "hand": list}, "player" : {"ace" : int, "hand": list}, "turn": int} ########

class player:
    def __init__(self, env, lr, batch_size, discount, epsilon, collect_period, update_period, decay_rate, decay_period, total_steps, use_ddqn):
        self.state = None
        self.use_ddqn = use_ddqn
        self.discount = torch.tensor(discount, dtype=torch.float32)
        self.env = env
        self.buffer = ReplayBuffer()
        self.batch_size = batch_size
        self.update_period = update_period
        self.collect_period = collect_period
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.decay_period = decay_period
        self.q_net = Qnet()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr = lr)
        self.t_net = Qnet()
        self.loss = nn.MSELoss()
        self.total_steps = total_steps
        

    def encode_state(self, state:dict) -> torch.tensor:
        dealer_card = state["dealer"]["hand"]       # list, elemement: ("suit" : str, rank: int)
        player_hand = state["player"]["hand"]       # list

        
        dealer_list = [dealer_card[0][1]]
        player_list = [card[1] for card in player_hand]
        num_hand = len(player_list)
        player_list = player_list + [0] * (5-num_hand)          # make in to 5 dimensional vector
        encoded_state = torch.tensor(dealer_list + player_list, dtype=torch.float32) 

        return encoded_state
    
    def collect_transitions(self) -> None:
        done = False

        state = self.env.reset()        # initialize
        
        while not done:
            encoded_state = self.encode_state(state)
            action = self.get_action(encoded_state)
            new_state, reward, done = self.env.step(state, action)

            encoded_state = self.encode_state(state)
            encoded_new_state = self.encode_state(new_state)
            
            interaction = (utils.tensor_to_numpy(encoded_state), action, utils.tensor_to_numpy(reward), utils.tensor_to_numpy(encoded_new_state))
            self.buffer.push(interaction)
            
            state = new_state
    
    def sample_trajectory(self) -> list:
        trajectory = []
        done = False

        state = self.env.reset()        # initialize
        
        while not done:
            encoded_state = self.encode_state(state)
            action = self.get_action(encoded_state)
            new_state, reward, done = self.env.step(state, action)

            encoded_state = self.encode_state(state)
            encoded_new_state = self.encode_state(new_state)

            trajectory.append((encoded_state, action, reward, encoded_new_state))

            state = new_state
        
        return trajectory
    
    def sample_random_trajectory(self) -> list:
        trajectory = []
        done = False

        state = self.env.reset()        # initialize
        
        while not done:
            encoded_state = self.encode_state(state)
            action = self.get_random_action(encoded_state)
            new_state, reward, done = self.env.step(state, action)

            encoded_state = self.encode_state(state)
            encoded_new_state = self.encode_state(new_state)

            trajectory.append((encoded_state, action, reward, encoded_new_state))

            state = new_state
        
        return trajectory
        


    def get_action(self, encoded_state: torch.tensor)-> np.ndarray:
        qa_values = self.get_qavalue(encoded_state, self.q_net)

        action = torch.argmax(qa_values).numpy()

        if np.random.rand() < self.epsilon:     # epsilon greedy policy
            action = 1 - action
        
        return action
    
    def get_random_action(self, encoded_state: torch.tensor) -> np.ndarray:
        return np.random.randint(0,2)


    def get_qavalue(self, encoded_state: torch.tensor, network: callable) -> torch.tensor:
        input = encoded_state.unsqueeze(0)
        qa_values = network(input)
        
        return qa_values.squeeze()
    
    def update_target_network(self) -> None:
        self.t_net.load_state_dict(self.q_net.state_dict())
        
    
    def update(self, state: torch.tensor, action: torch.tensor, reward: torch.tensor, next_state: torch.tensor) -> float:
        # update Q net with interaction, given as tensors
        if self.use_ddqn:
            target_qa_values = self.get_qavalue(next_state, self.t_net)
            qnet_qa_values = self.get_qavalue(next_state, self.q_net)
            max_index = torch.argmax(qnet_qa_values, dim=1)
            max_q_value = target_qa_values.gather(1, max_index.unsqueeze(1)).squeeze(1)
        else:
            next_qa_values = self.get_qavalue(next_state, self.t_net) # get q values from target network
            max_q_value, _ = torch.max(next_qa_values, dim = 1)

            
        target = reward + self.discount * max_q_value # target value

        self.optimizer.zero_grad()
        qa_values = self.get_qavalue(state, self.q_net)
        q_value = qa_values.gather(1, action.unsqueeze(1)).squeeze(1)   # get q value from q network

        loss = self.loss(q_value, target)
        loss.backward() #calculate gradient
        self.optimizer.step() #update

        return q_value.mean().item(), loss.item()


    def train(self) -> list:
        q_values = []
        losses = []
        for steps in range(self.total_steps):
            if (steps+1) % 1000 == 0:
                print("steps: {}".format(steps+1))
            if (steps+1) % self.decay_period == 0:
                self.epsilon *= self.decay_rate

            if steps % self.update_period == 0:
                self.update_target_network()
            
            if steps % self.collect_period == 0:
                for _ in range(50):                # collect transitions
                    self.collect_transitions()

            states, actions, rewards, next_states = self.buffer.sample(self.batch_size)

            mean_q_value, loss = self.update(states, actions, rewards, next_states)
            q_values.append(mean_q_value)
            losses.append(loss)
        return q_values, losses
    
    def test(self, is_random: bool) -> float:
        wins = 0
        num_rounds = 1000

        if is_random:
            for _ in range(num_rounds):
                trajectory = self.sample_random_trajectory()
                state, action, reward, new_state = trajectory[-1]
                
                wins += reward == 1
        else:
            for _ in range(num_rounds):
                trajectory = self.sample_trajectory()
                state, action, reward, new_state = trajectory[-1]
                
                wins += reward == 1

        return wins/num_rounds
