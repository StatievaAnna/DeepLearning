import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from wrapped_flappy_bird import GameState, SCREENHEIGHT, SCREENWIDTH

# os.environ['WANDB_MODE'] = 'offline'
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_episodes = 5000
max_steps_per_episode = 10000

game = GameState()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, error in zip(indices, td_errors):
            self.priorities[i] = abs(error) + 1e-5  # avoid zero

    def __len__(self):
        return len(self.buffer)



class FlappyBirdAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.replay_buffer = PrioritizedReplayBuffer(10000)

        self.batch_size = 64
        self.gamma = 0.99  # discount factor
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.update_target_every = 1000  # steps to update target network
        self.step_count = 0

        wandb.init(project="flappy-bird-dqn", name='', config={
            "episodes": num_episodes,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon,
            "epsilon_min":self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "lr": 1e-3,
        })

    def select_action(self, state):
        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_v = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_v)
            return q_values.argmax().item()

    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        current_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        td_errors = (expected_q_values - current_q_values).detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors)

        loss = (weights * (current_q_values - expected_q_values).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())



def get_state(game):
    upper_pipe_1 = game.upperPipes[0]
    lower_pipe_1 = game.lowerPipes[0]
    upper_pipe_2 = game.upperPipes[1] if len(game.upperPipes) > 1 else upper_pipe_1
    lower_pipe_2 = game.lowerPipes[1] if len(game.lowerPipes) > 1 else lower_pipe_1

    # Центр текущей щели
    gap_center_y = (upper_pipe_1['y'] + lower_pipe_1['y']) / 2
    dy_to_gap_center = (game.playery - gap_center_y) / SCREENHEIGHT

    state = np.array([
        game.playery / SCREENHEIGHT,
        game.playerVelY / 10,
        upper_pipe_1['x'] / SCREENWIDTH,
        upper_pipe_1['y'] / SCREENHEIGHT,
        lower_pipe_1['x'] / SCREENWIDTH,
        lower_pipe_1['y'] / SCREENHEIGHT,
        game.playerx / SCREENWIDTH,
        upper_pipe_2['x'] / SCREENWIDTH,
        upper_pipe_2['y'] / SCREENHEIGHT,
        lower_pipe_2['x'] / SCREENWIDTH,
        lower_pipe_2['y'] / SCREENHEIGHT,
        dy_to_gap_center
    ], dtype=np.float32)

    return state



# === Пример игрового цикла ===

agent = FlappyBirdAgent(state_dim=12, action_dim=2)

# num_episodes = 1000
# max_steps_per_episode = 10000

best_reward = -float('inf')  # начальное значение лучшей награды

for episode in range(num_episodes):
    game.__init__()  # сброс игры
    state = get_state(game)
    total_reward = 0

    for step in range(max_steps_per_episode):
        action_idx = agent.select_action(state)
        input_actions = [0, 0]
        input_actions[action_idx] = 1

        _, reward, done = game.frame_step(input_actions)

        if action_idx == 1:
            reward -= 0.1

        # Дополнительный штраф при столкновении — в зависимости от расстояния до центра щели 
        if done and reward == -1:
            gap_center = (game.upperPipes[0]['y'] + game.lowerPipes[0]['y']) / 2
            dy_to_gap_center = abs(game.playery - gap_center) / SCREENHEIGHT 
            reward -= dy_to_gap_center  # штраф до -2 при сильном отклонении

        total_reward += reward

        next_state = get_state(game)
        agent.push_experience(state, action_idx, reward, next_state, done)
        agent.train_step()

        state = next_state

        if done:
            # Уменьшать epsilon только если награда улучшилась
            if total_reward > best_reward:
                best_reward = total_reward
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

            print(f"Episode {episode + 1}, Steps: {step}, Total reward: {total_reward:.2f}, "
                  f"Best reward: {best_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Score: {game.score}, Best score: {game.bestScore}")

            break

    wandb.log({
        "episode": episode + 1,
        "reward": total_reward,
        "best_reward": best_reward,
        "score": game.score,
        "best_score": game.bestScore,
        "epsilon": agent.epsilon
    })
    

    