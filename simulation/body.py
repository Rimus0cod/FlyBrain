import torch
import math

class Simple2DBody:
    def __init__(self, batch_size=1, dt=0.05, device='cpu'):
        self.batch_size = batch_size
        self.dt = dt
        self.device = device
        
        # Физические константы тела
        self.mass = 1.0
        self.inertia = 0.5
        self.linear_drag = 0.1   # Трение о воздух (линейное)
        self.angular_drag = 0.2  # Трение вращения (чтобы не крутился вечно)
        self.max_thrust = 10.0   # Максимальная сила моторов
        
        self.reset()

    def reset(self):
        """Сброс состояния всех агентов в батче"""
        # [batch_size, 2] - позиции x, y
        self.pos = torch.zeros((self.batch_size, 2), device=self.device)
        self.vel = torch.zeros((self.batch_size, 2), device=self.device)
        
        # [batch_size, 1] - угол и угловая скорость
        # Инициализируем случайным углом от 0 до 2*PI, чтобы агенты учились выравниваться из любой позы
        self.angle = torch.rand((self.batch_size, 1), device=self.device) * 2 * math.pi
        self.ang_vel = torch.zeros((self.batch_size, 1), device=self.device)

    def step(self, motor_left, motor_right):
        """
        Шаг симуляции.
        motor_left, motor_right: тензоры формы [batch_size, 1] со значениями от 0 до 1.
        """
        # 1. Ограничиваем моторы (на случай если сеть выдаст бред)
        mL = torch.clamp(motor_left, 0.0, 1.0)
        mR = torch.clamp(motor_right, 0.0, 1.0)
        
        # 2. Вычисляем силы
        thrust = (mL + mR) * self.max_thrust
        torque = (mR - mL) * self.max_thrust
        
        # 3. Обновляем вращение (Угловое ускорение = Момент / Инерция)
        ang_acc = torque / self.inertia
        self.ang_vel = self.ang_vel + ang_acc * self.dt
        self.ang_vel = self.ang_vel * (1.0 - self.angular_drag) # Применяем трение
        self.angle = (self.angle + self.ang_vel * self.dt) % (2 * math.pi)
        
        # 4. Обновляем линейное движение
        # Тяга направлена туда, куда "смотрит" агент
        acc_x = thrust * torch.cos(self.angle) / self.mass
        acc_y = thrust * torch.sin(self.angle) / self.mass
        
        self.vel[:, 0] += acc_x[:, 0] * self.dt
        self.vel[:, 1] += acc_y[:, 0] * self.dt
        self.vel = self.vel * (1.0 - self.linear_drag) # Линейное трение
        
        self.pos += self.vel * self.dt
        
        return self.get_state()

    def get_state(self):
        """Возвращает текущую физику для передачи в сенсоры"""
        return {
            "pos": self.pos.clone(),
            "vel": self.vel.clone(),
            "angle": self.angle.clone(),
            "ang_vel": self.ang_vel.clone()
        }