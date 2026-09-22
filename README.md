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

В воспроизводимом эксперименте по сравнению базового решения с FlyBrain для обеих стратегий используются одни и те же параметры: двумерная среда, контракт наблюдений, функция вознаграждения, ограничения эпизода и бюджет PPO.
Запустите его из активированного виртуального окружения:

```bash
python -m training.run_experiment \
  --config configs/experiments/milestone_001_ppo.json
```

Результат обработки конфигурации, контрольную точку, метрики по каждой политике и сводные данные записьваются в каталог с меткой времени, расположенный по адресу... `experiments/runs/`. Метрики включают успех и
частота столкновений, эффективность пути, среднее вознаграждение, устойчивость управления, параметр
задержка подсчета и вывода. Результаты — это измерения, а не претензии на превосходство.

После стабильной навигации появляются признаки зрительного потока/вырисовывания, затем
адаптер более сложного 3D-тела. Сам интерфейс при мозге этого не меняется.
