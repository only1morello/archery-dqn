"""
Записывает видео ЭВОЛЮЦИИ агента — от случайной стрельбы до точного прицеливания.

Тренирует модель с нуля, периодически записывая кадры.
Результат: быстрое видео где видно как агент учится на глазах.

Запуск:
  pip install imageio imageio-ffmpeg
  python -m utils.recorder
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys, os, math, random, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from env.archery_env import ArcheryEnv
from env.physics import simulate_arrow
from agent.dqn import DQN
from agent.replay_buffer import ReplayBuffer


def render_frame(ax, env, traj_points, hit, episode, total_hits, total_shots,
                 epsilon, phase_label):
    """Рисует один кадр — тёмная тема, оригинальный арчер."""
    ax.clear()
    ax.set_facecolor('#0a0a0a')
    ax.axhspan(270, 300, color='#1a2e1a', alpha=0.8)

    # Лучник (оригинальные цвета)
    body = patches.Rectangle((env.archer_x - 5, env.archer_y - 30), 10, 30,
                               facecolor='#8B4513', edgecolor='#5C3317', linewidth=1.5)
    ax.add_patch(body)
    head = plt.Circle((env.archer_x, env.archer_y - 35), 5,
                       facecolor='#DEB887', edgecolor='#8B4513', linewidth=1)
    ax.add_patch(head)

    # Мишень
    for r, color in [(env.target_radius, '#FF0000'),
                     (env.target_radius * 0.7, '#FFFFFF'),
                     (env.target_radius * 0.4, '#FF0000'),
                     (env.target_radius * 0.15, '#FFD700')]:
        circle = plt.Circle((env.target_x, env.target_y), r,
                            facecolor=color, edgecolor='#444444', linewidth=0.5)
        ax.add_patch(circle)

    # Траектория
    if traj_points:
        xs = [p[0] for p in traj_points if 0 <= p[0] <= 400 and 0 <= p[1] <= 300]
        ys = [p[1] for p in traj_points if 0 <= p[0] <= 400 and 0 <= p[1] <= 300]
        if len(xs) > 1:
            color = '#00FF44' if hit else '#FF4444'
            ax.plot(xs, ys, '-', color=color, linewidth=2.5, alpha=0.9)
            ax.plot(xs[-1], ys[-1], 'o', color=color, markersize=6, zorder=10)

    # HIT
    if hit:
        ax.text(env.target_x, env.target_y - 35, 'HIT!',
               fontsize=14, fontweight='bold', color='#00FF00', ha='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0a0a', alpha=0.8))

    # Инфо сверху
    hr = total_hits / max(total_shots, 1) * 100
    info = f'Ep {episode:,}  |  Hits: {total_hits}/{total_shots} ({hr:.0f}%)  |  ε={epsilon:.2f}'
    ax.text(200, 12, info, fontsize=9, color='#e0e0e0', ha='center',
           fontweight='bold', family='monospace',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', alpha=0.9))

    # Фаза обучения снизу
    ax.text(200, 288, phase_label, fontsize=11, color='#e0e0e0', ha='center',
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', alpha=0.9))

    ax.set_xlim(0, 400)
    ax.set_ylim(300, 0)
    ax.set_aspect('equal')
    ax.axis('off')


def get_phase_label(episode):
    if episode < 1000:
        return "PHASE 1 — Random Shooting"
    elif episode < 4000:
        return "PHASE 2 — Exploring"
    elif episode < 8000:
        return "PHASE 3 — Learning to Aim"
    elif episode < 14000:
        return "PHASE 4 — Getting Accurate"
    else:
        return "PHASE 5 — Trained Agent"


def record_evolution(save_path='agent_evolution.mp4',
                     total_episodes=20000,
                     record_every=50,
                     fps=24):
    """
    Тренирует с нуля и записывает видео по ходу.

    total_episodes: сколько эпизодов тренировать
    record_every: записывать кадр каждые N эпизодов
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("Установи: pip install imageio imageio-ffmpeg")
        return

    env = ArcheryEnv()
    inp = 30 * 30
    policy = DQN(inp, 32, 24, env.num_actions, lr=5e-4)
    target = DQN(inp, 32, 24, env.num_actions, lr=5e-4)
    target.copy_weights_from(policy)
    buf = ReplayBuffer(50000)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor('#0a0a0a')
    writer = imageio.get_writer(save_path, fps=fps, quality=8)

    total_hits = 0
    total_shots = 0
    recent_hits = []
    start = time.time()

    print(f"Training {total_episodes:,} episodes, recording every {record_every}...")
    print(f"Estimated video: ~{total_episodes // record_every / fps:.0f} seconds at {fps}fps")

    for ep in range(total_episodes):
        epsilon = 0.05 + 0.95 * math.exp(-ep / 3000)

        frame = env.reset()

        # Выбор действия
        if random.random() < epsilon:
            action = random.randrange(env.num_actions)
        else:
            flat = frame.flatten().astype(np.float32).reshape(1, -1) / 255.0
            q = policy.forward(flat)
            action = int(np.argmax(q[0]))

        angle, force = env.actions[action]
        nf, reward, done, info = env.step(action)
        hit = info['hit']

        # Буфер + обучение
        sf = frame.flatten().astype(np.float32) / 255.0
        nff = nf.flatten().astype(np.float32) / 255.0
        buf.push(sf, action, reward, nff, 1.0)

        if len(buf) >= 500:
            st, ac, rw, ns, dn = buf.sample(32)
            cq = policy.forward(st)
            nq = target.forward(ns)
            er = np.clip(cq[np.arange(32), ac] - (rw + 0.99 * np.max(nq, 1) * (1 - dn)), -1, 1)
            do = np.zeros_like(cq)
            do[np.arange(32), ac] = er
            policy.backward(do)
            if ep % 500 == 0:
                target.copy_weights_from(policy)

        if hit:
            total_hits += 1
        total_shots += 1
        recent_hits.append(1.0 if hit else 0.0)

        # Записываем кадр
        if ep % record_every == 0:
            traj = simulate_arrow(env.archer_x, env.archer_y, angle, force)
            phase = get_phase_label(ep)

            render_frame(ax, env, traj, hit, ep, total_hits, total_shots,
                        epsilon, phase)

            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            writer.append_data(image)

        # Лог
        if (ep + 1) % 2000 == 0:
            hr = np.mean(recent_hits[-1000:]) * 100
            elapsed = time.time() - start
            frames_done = ep // record_every
            print(f"  Ep {ep+1:6,} | Hit: {hr:5.1f}% | ε: {epsilon:.3f} | "
                  f"Frames: {frames_done} | Time: {elapsed:.0f}s")

    # Финальные кадры — отдельный счётчик, больше выстрелов
    print("Recording final showcase (slow)...")
    final_hits = 0
    final_shots = 0
    for _ in range(30):
        frame = env.reset()
        flat = frame.flatten().astype(np.float32).reshape(1, -1) / 255.0
        q = policy.forward(flat)
        action = int(np.argmax(q[0]))
        angle, force = env.actions[action]
        env.step(action)

        traj = simulate_arrow(env.archer_x, env.archer_y, angle, force)
        min_d = min(math.sqrt((x - env.target_x)**2 + (y - env.target_y)**2) for x, y in traj)
        hit = min_d <= env.target_radius
        if hit:
            final_hits += 1
        final_shots += 1

        render_frame(ax, env, traj, hit, total_episodes, final_hits, final_shots,
                    0.05, "TRAINED AGENT — Final Performance")
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        for _ in range(int(fps * 0.5)):
            writer.append_data(image)

    writer.close()
    plt.close()

    duration = (total_episodes // record_every + 30 * int(fps * 0.5)) / fps
    print(f"\nSaved: {save_path}")
    print(f"Duration: ~{duration:.0f}s")
    print(f"Training hit rate: {np.mean(recent_hits[-1000:])*100:.1f}%")
    print(f"Final showcase: {final_hits}/{final_shots} ({final_hits/final_shots*100:.0f}%)")


if __name__ == '__main__':
    record_evolution()
