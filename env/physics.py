import math
def simulate_arrow(x0, y0, angle_deg, force, gravity=9.8, steps=40):
    """В общем, возвращает список точек [(x, y), (x, y), ... ]
        траекторию стрелы от (x0, y0) с заданным углом и силой.

    angle_deg: угол в градусах от горизонтали (вверх)
    force: начальная скорость
    gravity: ускорение свободного падения (g)
    steps: количество шагов симуляции
    """
    # п рад = 180 град.
    radian = 180 / math.pi # если в одной радиане 180/P = примерно 57.324 градусов
    angle_rad = angle_deg / radian
    
    x_velocity = force * math.cos(angle_rad) # constant
    y_velocity = force * math.sin(angle_rad) # но гравитация тянет вниз
    dt = 0.15 # each step

# Скорость по оси x постоянна (если не учитывать сопротивление воздуха), поэтому в каждом кадре стрела будет перемещаться на одинаковое расстояние по направлению x.
# X(t) = x0 + x_velocity * t

# На скорость по оси y влияет гравитация, поэтому в каждом кадре она будет тянуться все дальше вниз.

# Y(t) = y0 - v_y * t + 0.5 * g * t2
# Xo и Yo обозначают начальное положение.
# лучник примерно на (30, 250) — внизу слева, а мишень где-то на (350, 150) — справа.

    pos_list = []

    for i in range(steps + 1):
        t = i * dt # time
        x_pos = x0 + x_velocity * t
        y_pos = y0 - y_velocity * t + 0.5 * gravity * t**2
        pos_list.append((x_pos, y_pos))
    
    return pos_list
    

