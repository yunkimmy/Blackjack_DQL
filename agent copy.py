import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network import Qnet

#### state = {"dealer" : {"ace" : int, "hand": list}, "player" : {"ace" : int, "hand": list}, "turn": int} ########

class player:
    def __init__(self, env, lr, batch_size, discount, epsilon, update_period, decay_rate, decay_period, total_steps, use_ddqn):
        self.state = None
        self.use_ddqn = use_ddqn
        self.discount = torch.tensor(discount, dtype=torch.float32)
        self.env = env
        self.batch = batch_size
        self.update_period = update_period
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


    def get_action(self, encoded_state: torch.tensor)-> int:
        qa_values = self.get_qavalue(encoded_state, self.q_net)

        action = torch.argmax(qa_values).item()

        if np.random.rand() < self.epsilon:     # epsilon greedy policy
            action = 1 - action
        
        return action
    
    def get_random_action(self, encoded_state: torch.tensor) -> int:
        return np.random.randint(0,2)


    def get_qavalue(self, encoded_state: torch.tensor, network: callable) -> torch.tensor:
        input = encoded_state.unsqueeze(0)
        qa_values = network(input)
        
        return qa_values.squeeze()
    
    def update_target_network(self) -> None:
        self.t_net.load_state_dict(self.q_net.state_dict())
        
    
    def update(self, trajectory: list) -> float:     # update Q net with trajectory
        q_values = []
        for interaction in trajectory:
            state, action, reward, next_state = interaction
            if self.use_ddqn:
                target_qa_values = self.get_qavalue(next_state, self.t_net)
                qnet_qa_values = self.get_qavalue(next_state, self.q_net)
                max_index = torch.argmax(qnet_qa_values)
                max_q_value = target_qa_values[max_index]
            else:
                next_qa_values = self.get_qavalue(next_state, self.t_net) # get q values from target network
                max_q_value = torch.max(next_qa_values)
            
            target = reward + self.discount * max_q_value # target value

            self.optimizer.zero_grad()
            qa_values = self.get_qavalue(state, self.q_net)
            q_value = qa_values[action]     # get q value from q network
            q_values.append(q_value.detach())
            loss = self.loss(q_value, target)
            loss.backward() #calculate gradient
            self.optimizer.step() #update
        return np.mean(q_values)


    def train(self) -> list:
        q_values = []
        for steps in range(self.total_steps):
            if (steps+1) % 5000 == 0:
                print("steps: {}".format(steps+1))
            if (steps+1) % self.decay_period == 0:
                self.epsilon *= self.decay_rate

            if steps % self.update_period == 0:
                self.update_target_network()

            trajectory = self.sample_trajectory()
            mean_q_value = self.update(trajectory)
            q_values.append(mean_q_value)
        return q_values
    
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
