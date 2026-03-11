import numpy as np
import random
import math
from env.physics import simulate_arrow

class ArcheryEnv:
    def __init__(self):
        self.width = 400
        self.height = 300
        self.archer_x = 30
        self.archer_y = 250    # внизу слева
        self.target_radius = 20

        # таблица действий: action_id -> (angle, force)
        # 4 угла × 3 силы = 12 действий
        self.actions = []
        for angle in [20, 35, 50, 65]:
            for force in [80, 120, 160]:
                self.actions.append((angle, force))
        self.num_actions = len(self.actions)

        # состояние
        self.target_x = 350
        self.target_y = 150
        self.arrow_trajectory = None
        self.arrow_hit = False

    def reset(self):
        # мишень справа, на случайной высоте
        self.target_x = 350
        self.target_y = random.randint(50, 250)  # случайная высота
        self.arrow_trajectory = None
        self.arrow_hit = False

        return self._get_frame()

    def step(self, action):
        # action -> (angle, force) из таблицы
        angle, force = self.actions[action]

        # запустить стрелу (simulate_arrow)
        trajectory = simulate_arrow(self.archer_x, self.archer_y, angle, force)
        self.arrow_trajectory = trajectory

        # проверить попадание — ищем ближайшую точку траектории к мишени
        min_distance = float('inf')
        for (x, y) in trajectory:
            dist = math.sqrt((x - self.target_x) ** 2 + (y - self.target_y) ** 2)
            if dist < min_distance:
                min_distance = dist

        # премия сотруднику
        if min_distance <= self.target_radius:
            reward = 10.0
            self.arrow_hit = True
        elif min_distance <= self.target_radius + 30:
            reward = 2.0
        else:
            reward = -1.0

        done = True
        info = {"distance": min_distance, "hit": self.arrow_hit,
                "angle": angle, "force": force}

        return self._get_frame(), reward, done, info

    def _get_frame(self):
        # рисуем кадр в numpy массив
        canvas = np.ones((self.height, self.width), dtype=np.uint8) * 200  # небо

        # земля
        canvas[270:, :] = 80

        # лучник (прямоугольник)
        canvas[self.archer_y - 30 : self.archer_y,
               self.archer_x - 5 : self.archer_x + 5] = 40

        # мишень (круг через маску расстояний)
        yy, xx = np.ogrid[:self.height, :self.width]
        mask = (xx - self.target_x)**2 + (yy - self.target_y)**2 <= self.target_radius**2
        canvas[mask] = 0

        # траектория стрелы
        if self.arrow_trajectory is not None:
            for (x, y) in self.arrow_trajectory:
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    y1, y2 = max(0, iy - 1), min(self.height, iy + 2)
                    x1, x2 = max(0, ix - 1), min(self.width, ix + 2)
                    canvas[y1:y2, x1:x2] = 255

        # уменьшаем до 84x84 !update guys - 30x30 
        rows = np.linspace(0, self.height - 1, 30).astype(int)
        cols = np.linspace(0, self.width - 1, 30).astype(int)
        small = canvas[np.ix_(rows, cols)]

        return small  # (30, 30), uint8

    def close(self):
        pass
