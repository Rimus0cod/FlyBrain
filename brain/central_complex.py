"""Recurrent heading representation used by the fly-inspired controller."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralComplex(nn.Module):
    def __init__(self, num_neurons=16, sensory_input_dim=2):
        super().__init__()
        self.num_neurons = num_neurons
        
        # Входные веса: переводят сырые сенсоры (например, [angular_velocity, visual_flow]) 
        # в стимуляцию кольца
        self.W_in = nn.Linear(sensory_input_dim, num_neurons, bias=False)
        
        # Рекуррентные веса кольца
        # В биологии связи фиксированы эволюцией: соседи возбуждают друг друга, дальние - тормозят.
        # Мы создаем обучаемую матрицу (parameter), но инициализируем её биологически логично.
        self.W_rec = nn.Parameter(self._init_ring_weights())
        
        # Внутреннее состояние (компас)
        self.state = None 

    def _init_ring_weights(self):
        """
        Инициализация весов 'Мексиканская шляпа' (локальное возбуждение, глобальное торможение).
        Это позволяет пику активности стабильно существовать на кольце.
        """
        W = torch.zeros((self.num_neurons, self.num_neurons))
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                # Вычисляем кратчайшее расстояние по кольцу
                dist = min(abs(i - j), self.num_neurons - abs(i - j))
                # Соседи (+1), чуть дальше (-0.5), далеко (-0.1)
                if dist == 0: W[i, j] = 1.0
                elif dist == 1: W[i, j] = 0.5
                else: W[i, j] = -0.2
        return W

    def reset_state(self, batch_size=1, device=None):
        """Сброс состояния при начале нового эпизода симуляции"""
        # Начинаем с нейтрального состояния (нули) или случайного шума
        device = device or self.W_rec.device
        self.state = torch.zeros(batch_size, self.num_neurons, device=device)
        # Можно искусственно возбудить нулевой нейрон (стартуем смотря на 0 градусов)
        self.state[:, 0] = 1.0

    def forward(self, sensory_input):
        """
        Обновление состояния (f(state[t], input[t]))
        sensory_input: тензор формы [batch_size, sensory_input_dim]
        """
        if self.state is None or self.state.shape[0] != sensory_input.shape[0]:
            self.reset_state(
                batch_size=sensory_input.shape[0], device=sensory_input.device
            )
            
        # 1. Влияние сенсоров (например, поворот тела)
        stimulus = self.W_in(sensory_input)
        
        # 2. Рекуррентное влияние кольца на само себя (поддержание пика)
        # Умножаем текущее состояние на матрицу весов
        ring_effect = torch.matmul(self.state, self.W_rec)
        
        # 3. Новое состояние (с нелинейностью для отсечения отрицательной активности)
        new_state = F.relu(ring_effect + stimulus)
        
        # Опционально: нормализация, чтобы активность не взрывалась до бесконечности
        new_state = new_state / (new_state.sum(dim=-1, keepdim=True) + 1e-8)
        
        self.state = new_state
        return self.state
