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

## Milestone 001 PPO comparison

The reproducible baseline-vs-FlyBrain experiment uses the same 2D environment,
observation contract, reward, episode limits and PPO budget for both policies.
Run it from the activated virtual environment:

```bash
python -m training.run_experiment \
  --config configs/experiments/milestone_001_ppo.json
```

It writes the resolved config, checkpoint, per-policy metrics and a summary to a
timestamped directory under `experiments/runs/`. Metrics include success and
collision rates, path efficiency, mean reward, control stability, parameter
count and inference latency. Results are measurements, not claims of superiority.

После стабильной навигации добавляются препятствия, visual-flow/looming, затем
адаптер более сложного 3D-тела. Сам интерфейс мозга при этом не меняется.
