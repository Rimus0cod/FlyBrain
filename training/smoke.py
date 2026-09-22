"""Run one episode step through either controller to validate the wiring."""

import torch

from brain import BaselineMLP, FlyBrainController
from simulation.environment import Navigation2DEnvironment


def run_smoke_test() -> None:
    environment = Navigation2DEnvironment(batch_size=2)
    observation = environment.reset()
    flybrain = FlyBrainController(visual_dim=4)
    baseline = BaselineMLP(observation_dim=observation.shape[-1])

    flybrain_state = flybrain.initial_state(batch_size=observation.shape[0])
    for controller in (baseline, flybrain):
        if controller is flybrain:
            motors, flybrain_state = controller(observation, flybrain_state)
        else:
            motors = controller(observation)
        _, reward, terminated, truncated, info = environment.step(motors)
        assert motors.shape == (2, 2)
        assert reward.shape == terminated.shape == truncated.shape == info["distance"].shape == (2,)


if __name__ == "__main__":
    run_smoke_test()
    print("Smoke test passed.")
