"""
Главный скрипт обучения.
Запуск: python train.py
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.archery_env import ArcheryEnv
from agent.trainer import DQNTrainer

NUM_EPISODES = 20000
LOG_EVERY = 200
SAVE_EVERY = 2000


def train():
    env = ArcheryEnv()
    trainer = DQNTrainer(num_actions=env.num_actions, lr=5e-4)

    metrics = {"episodes": [], "rewards": [], "epsilons": [], "losses": [], "hit_rates": []}
    recent_rewards = []
    recent_hits = []

    print(f"Обучение DQN: {NUM_EPISODES} эпизодов")
    print(f"Действий: {env.num_actions}")
    print(f"Epsilon: {trainer.eps_start} → {trainer.eps_end}")
    print("-" * 60)

    for episode in range(NUM_EPISODES):
        trainer.episode = episode

        # 1. Новый эпизод
        frame = env.reset()  # (84, 84) uint8

        # 2. Агент выбирает действие
        action = trainer.select_action(frame)

        # 3. Среда выполняет выстрел
        next_frame, reward, done, info = env.step(action)

        # 4. Запоминаем (храним flat + normalized)
        state_flat = frame.flatten().astype(np.float32) / 255.0
        next_flat = next_frame.flatten().astype(np.float32) / 255.0
        trainer.buffer.push(state_flat, action, reward, next_flat, float(done))

        # 5. Обучаем
        loss = trainer.train_step()

        # Статистика
        recent_rewards.append(reward)
        recent_hits.append(1.0 if info["hit"] else 0.0)

        if (episode + 1) % LOG_EVERY == 0:
            avg_reward = np.mean(recent_rewards[-LOG_EVERY:])
            hit_rate = np.mean(recent_hits[-LOG_EVERY:]) * 100
            eps = trainer.get_epsilon()
            loss_str = f"{loss:.4f}" if loss is not None else "waiting"

            print(f"Ep {episode+1:6d} | "
                  f"Reward: {avg_reward:+5.2f} | "
                  f"Hit: {hit_rate:5.1f}% | "
                  f"Eps: {eps:.3f} | "
                  f"Loss: {loss_str} | "
                  f"Buf: {len(trainer.buffer)}")

            metrics["episodes"].append(episode + 1)
            metrics["rewards"].append(float(avg_reward))
            metrics["hit_rates"].append(float(hit_rate))
            metrics["epsilons"].append(float(eps))
            metrics["losses"].append(float(loss) if loss else 0)

        if (episode + 1) % SAVE_EVERY == 0:
            os.makedirs("checkpoints", exist_ok=True)
            trainer.save(f"checkpoints/dqn_ep{episode+1}.npz")

    # Сохраняем финал
    os.makedirs("checkpoints", exist_ok=True)
    trainer.save("checkpoints/dqn_final.npz")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    print("-" * 60)
    final_hit = np.mean(recent_hits[-500:]) * 100
    print(f"Готово! Финальный hit rate: {final_hit:.1f}%")
    env.close()
    return metrics


if __name__ == "__main__":
    train()
