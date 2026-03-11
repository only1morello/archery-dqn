import numpy as np
import random
import math

from agent.dqn import DQN
from agent.replay_buffer import ReplayBuffer


"""
DQN Trainer — здесь происходит обучение-головокружение

crucial concepts:

1. epsilon-greedy:
   С вероятностью e — случайное действие (exploration).
   С вероятностью (1-e) — лучшее по Q (exploitation).
   e падает с 1.0 до 0.05 за ~5000 эпизодов.

2. target-network:
   Две копии сети: policy_net (учится) и target_net (замороженная).
   target_net даёт стабильную "цель" для обучения.
   Синхронизация каждые 500 шагов.

3. Беллманская формула:
   target_Q = reward + gamma * max(target_net(next_state))
   "Ценность = награда сейчас + лучшая будущая ценность"
"""

class DQNTrainer:
    def __init__(self, num_actions=12, lr=3e-4):
        self.num_actions = num_actions
        self.input_size = 30 * 30  # один кадр, flattened

        # Две сети
        self.policy_net = DQN(input_size=self.input_size, num_actions=num_actions, lr=lr)
        self.target_net = DQN(input_size=self.input_size, num_actions=num_actions, lr=lr)
        self.target_net.copy_weights_from(self.policy_net)

        # Буфер
        self.buffer = ReplayBuffer(capacity=50000)

        # Гиперпараметры
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 3000
        self.target_sync = 500
        self.min_buffer = 1000

        # Счётчики
        self.total_steps = 0
        self.episode = 0

    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-self.episode / self.eps_decay)

    def select_action(self, state):
        """
        state: numpy (30, 30) — один кадр
        """
        if random.random() < self.get_epsilon():
            return random.randrange(self.num_actions)
        else:
            flat = state.flatten().astype(np.float32).reshape(1, -1) / 255.0
            q_values = self.policy_net.forward(flat)  # (1, 12)
            return int(np.argmax(q_values[0]))

    def train_step(self):
        if len(self.buffer) < self.min_buffer:
            return None

        # 1. Случайный батч из буфера
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # states: (32, 7056) — уже flat и normalized при push

        # 2. Q-значения для текущих состояний
        current_q_all = self.policy_net.forward(states)  # (32, 12)
        # Выбираем Q только для тех действий, которые были сделаны
        current_q = current_q_all[np.arange(self.batch_size), actions]  # (32,)

        # 3. Target Q по формуле Беллмана
        next_q_all = self.target_net.forward(next_states)  # (32, 12)
        next_q_max = np.max(next_q_all, axis=1)  # (32,) — лучшее Q для следующего состояния
        target_q = rewards + self.gamma * next_q_max * (1 - dones)

        # 4. Градиент ошибки (Huber-подобный: обрезаем большие ошибки)
        error = current_q - target_q  # (32,)
        error = np.clip(error, -1.0, 1.0)  # clip = аналог Huber loss

        # Преобразуем градиент обратно в форму (32, 12)
        # Ненулевой градиент только для выбранных действий
        dout = np.zeros_like(current_q_all)  # (32, 12)
        dout[np.arange(self.batch_size), actions] = error

        # 5. Backprop + обновление весов
        self.policy_net.backward(dout)

        loss = float(np.mean(error ** 2))

        # 6. Синхронизация target network
        self.total_steps += 1
        if self.total_steps % self.target_sync == 0:
            self.target_net.copy_weights_from(self.policy_net)

        return loss

    def save(self, path):
        self.policy_net.save(path)

    def load(self, path):
        self.policy_net.load(path)
        self.target_net.copy_weights_from(self.policy_net)
