import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
import json
import os
import datetime

class Agent:
    def __init__(self, params=None):
        if params is None:
            params = {}
        
        self.params = params
        self.n_games = 0
        self.epsilon = params.get('epsilon_start', 80)
        self.epsilon_decay = params.get('epsilon_decay', 1)
        self.epsilon_min = params.get('epsilon_min', 0)
        self.gamma = params.get('gamma', 0.9)
        self.batch_size = params.get('batch_size', 1000)
        self.memory = deque(maxlen=params.get('replay_memory_size', 100000))
        self.lr = params.get('learning_rate', 0.001)
        self.target_update_freq = params.get('target_update_freq', 10)
        
        # Models
        self.model = Linear_QNet(11, 256, 3)
        self.target_model = Linear_QNet(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.trainer = QTrainer(self.model, self.target_model, lr=self.lr, gamma=self.gamma)
        self.last_q_values = [0, 0, 0]

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, game):
        return game._get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
            
        if not mini_sample:
            return 0.0

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        if self.n_games % self.target_update_freq == 0:
            self.update_target_network()
            
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, evaluate=False):
        if not evaluate:
            self.epsilon = max(self.epsilon_min, self.params.get('epsilon_start', 80) - (self.n_games * self.epsilon_decay))
        
        final_move = [0,0,0]
        state0 = torch.tensor(np.array(state), dtype=torch.float)
        prediction = self.model(state0)
        self.last_q_values = prediction.detach().numpy().tolist()
        
        if not evaluate and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self, model_name, score, avg_score):
        folder = './models'
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        base_name = f"{model_name}_{self.n_games}ep"
        model_path = os.path.join(folder, f"{base_name}.pth")
        meta_path = os.path.join(folder, f"{base_name}.json")
        
        torch.save(self.model.state_dict(), model_path)
        
        metadata = {
            "name": model_name,
            "date": datetime.datetime.now().isoformat(),
            "episodes": self.n_games,
            "best_score": score,
            "average_score": avg_score,
            "hyperparameters": self.params
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()
