# FlyBrain

Биоинспирированный проект автономной навигации на основе вычислительных принципов нервной системы *Drosophila melanogaster*.

## Документация

- `docs/architecture.md` — общая архитектура и границы модулей.
- `docs/sensory-model.md` — зрение, optic-flow abstraction, looming, olfaction, IMU/halteres.
- `docs/central-complex.md` — recurrent heading model и ring-attractor abstraction.
- `docs/mushroom-body.md` — sparse associative memory.
- `docs/reward-design.md` — RL-задачи, reward и защита от reward hacking.
- `docs/experiments.md` — baselines, ablations, benchmarks и Sim2Real protocol.

## Первый milestone — 2D proof of concept

Первый эксперимент намеренно отделён от физики квадрокоптера:

```text
Simple2DBody -> Simple2DSensors -> [Baseline MLP | FlyBrain] -> 2 motors
```

- `BaselineMLP` и `FlyBrainController` получают один и тот же вектор наблюдений и
  возвращают два нормализованных моторных сигнала.
- `FlyBrainController` реализует `Vision -> Central Complex -> Motor`; кольцевое
  состояние сбрасывается в начале каждого эпизода.
- Цель используется средой для формирования слабого визуального признака и reward,
  но её мировые координаты не передаются контроллеру.
- Подключение PPO/другого RL-раннера должно жить в `training/`, а результаты и
  конфигурации экспериментов — в `experiments/`.

Проверить связность каркаса можно командой:

```bash
pip install -r requirements.txt
python -m training.smoke
```

После стабильной навигации добавляются препятствия, visual-flow/looming, затем
адаптер более сложного 3D-тела. Сам интерфейс мозга при этом не меняется.
