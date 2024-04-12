import mo_gymnasium as mo_gym
import numpy as np

from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.multi_policy.morld.morld import MORLD


def main():
    gamma = 0.98
    def make_env():
        env = mo_gym.make("minecart-deterministic-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        return env

    env = make_env()
    eval_env = make_env()

    algo = MORLD(
        env=env,
        exchange_every=int(1e3),
        pop_size=10,
        policy_name="EUPG",
        scalarization_method="tch",
        evaluation_mode="esr",
        gamma=gamma,
        log=True,
        neighborhood_size=1,
        update_passes=10,
        shared_buffer=False,
        sharing_mechanism=[],
        weight_adaptation_method=None,
        project_name="MORL-Baselines",
        experiment_name="MORL-D",
    )

    algo.train(
        eval_env=eval_env,
        total_timesteps=int(1e6) + 1,
        ref_point=np.array([-1, -1, -200.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.98),
    )


if __name__ == "__main__":
    main()
