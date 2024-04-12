import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
# from gymnasium.wrappers.record_video import RecordVideo


def main(algo: str="gpi-ls", g: int = 10, timesteps_per_iter: int = 10000):
    def make_env():
        env = mo_gym.make("deep-sea-treasure-v0")
        env = mo_gym.MORecordEpisodeStatistics(env, gamma=0.99)
        return env

    env = make_env()
    eval_env = make_env()  # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = GPILS(
        env,
        num_nets=2,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(1e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        learning_starts=100,
        alpha_per=0.6,
        min_priority=0.01,
        per=True,
        gradient_updates=g,
        target_net_update_freq=200,
        tau=1,
        dynamics_uncertainty_threshold=1.5,
        dynamics_net_arch=[256, 256, 256],
        dynamics_buffer_size=int(1e5),
        dynamics_rollout_batch_size=25000,
        dynamics_train_freq=lambda t: 250,
        dynamics_rollout_freq=250,
        dynamics_rollout_starts=5000,
        dynamics_rollout_len=1,
        real_ratio=0.5,
        log=True,
        project_name="MORL-Baselines",
    )

    agent.train(
        total_timesteps=100 * timesteps_per_iter,
        eval_env=eval_env,
        ref_point=np.array([0.0, -50.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
        weight_selection_algo=algo,
        timesteps_per_iter=timesteps_per_iter,
    )


if __name__ == "__main__":
    main()
