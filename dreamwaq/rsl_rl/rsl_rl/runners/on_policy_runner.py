# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# Modified by Jungyeon Lee (curieuxjy) for DreamWaQ implementation
# Added OnPolicyRunnerWaq and OnPolicyRunnerEst classes
# https://github.com/curieuxjy

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

from rsl_rl.vae import CENet, EstNet
from rsl_rl.utils import RunningMeanStd


class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.oracle = True if self.cfg["run_name"] == "oracle" else False

        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        env_cfg = self.env.cfg.env  # A1RoughCfg.env

        # set critic observation dimension
        if self.env.num_privileged_obs is not None:
            num_critic_obs = env_cfg.num_observations + env_cfg.num_privileged_obs
        else:
            num_critic_obs = env_cfg.num_observations

        # set actor observation dimension
        if self.oracle:
            num_actor_obs = env_cfg.num_observations + env_cfg.num_privileged_obs
        else:
            num_actor_obs = env_cfg.num_observations

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            num_actor_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.rms_dict = {}
        if self.cfg["obs_rms"]:  # Initialize later
            self.obs_rms = None
        if self.cfg["privileged_obs_rms"]:  # Initialize later
            self.privileged_obs_rms = None

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # initial state
        obs = self.env.get_observations()
        if self.cfg["obs_rms"]:
            if self.obs_rms is None:
                self.obs_rms = RunningMeanStd(shape=obs.shape[1], device=self.device)
            self.obs_rms.update(obs.detach())
            obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)

        privileged_obs = self.env.get_privileged_observations()
        if self.oracle:
            if self.cfg["privileged_obs_rms"]:
                if self.privileged_obs_rms is None:
                    self.privileged_obs_rms = RunningMeanStd(
                        shape=privileged_obs.shape[1], device=self.device
                    )
                self.privileged_obs_rms.update(privileged_obs.detach())
                privileged_obs = (
                    privileged_obs - self.privileged_obs_rms.mean
                ) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

        # critic_obs = privileged_obs if privileged_obs is not None else obs
        critic_obs = (
            torch.cat((obs, privileged_obs), dim=-1)
            if privileged_obs is not None
            else obs
        )
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rew_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    actor_obs = critic_obs if self.oracle else obs

                    actions = self.alg.act(actor_obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)

                    if self.cfg["obs_rms"]:
                        self.obs_rms.update(obs.detach())
                        obs = (obs - self.obs_rms.mean.to(self.device)) / torch.sqrt(
                            self.obs_rms.var.to(self.device) + 1e-8
                        )

                    if self.oracle:
                        privileged_obs = privileged_obs.to(self.device)
                        if self.cfg["privileged_obs_rms"]:
                            self.privileged_obs_rms.update(privileged_obs.detach())
                            privileged_obs = (
                                privileged_obs
                                - self.privileged_obs_rms.mean.to(self.device)
                            ) / torch.sqrt(
                                self.privileged_obs_rms.var.to(self.device) + 1e-8
                            )

                    critic_obs = (
                        torch.cat((obs, privileged_obs), dim=-1)
                        if privileged_obs is not None
                        else obs
                    )

                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        if "reward_cv" in infos:
                            rew_infos.append(infos["reward_cv"])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:

                if self.cfg["obs_rms"]:
                    self.rms_dict["obs_rms"] = self.obs_rms
                if self.cfg["privileged_obs_rms"]:
                    self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms

                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()
            rew_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        info_types = [("ep_infos", "Episode"), ("rew_infos", "CV")]

        for info_type, prefix in info_types:
            if locs[info_type]:
                for key in locs[info_type][0]:  # one agent
                    infotensor = torch.tensor([], device=self.device)
                    for info in locs[info_type]:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(info[key], torch.Tensor):
                            info[key] = torch.Tensor([info[key]])
                        if len(info[key].shape) == 0:
                            info[key] = info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, info[key].to(self.device)))

                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f"{prefix}/{key}", value, locs["it"])

                    if prefix == "Episode":
                        ep_string += (
                            f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                        )

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
                "rms": self.rms_dict,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        if self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"]:
            self.rms_info = loaded_dict["rms"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_rms(self):
        return (
            self.rms_info
            if (self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"])
            else None
        )


class OnPolicyRunnerWAQ:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.vae_cfg = train_cfg["vae"]
        self.device = device
        self.env = env  # LeggedRobot

        env_cfg = self.env.cfg.env  # A1RoughWaqCfg.env
        num_critic_obs = (
            env_cfg.num_observations + env_cfg.num_estvel + env_cfg.num_privileged_obs
        )
        num_actor_obs = (
            env_cfg.num_observations + env_cfg.num_estvel + env_cfg.num_context
        )

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            num_actor_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        vae_class = eval(self.cfg["vae_class_name"])  # CENet
        self.cenet: CENet = vae_class(device=self.device, **self.vae_cfg).to(
            self.device
        )

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        self.cenet.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [env_cfg.len_obs_history * env_cfg.num_observations],
            [env_cfg.num_estvel],
            [env_cfg.num_observations],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.rms_dict = {}
        if self.cfg["obs_rms"]:  # Initialize later
            self.obs_rms = None
        if self.cfg["privileged_obs_rms"]:  # Initialize later
            self.privileged_obs_rms = None
        if self.cfg["true_vel_rms"]:
            self.true_vel_rms = None

        _, _ = self.env.reset()

    #? DreamWaQ Main Learning Loop
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.cenet.train_mode()

        ep_infos = []
        rew_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # O_{t}
            obs = self.env.get_observations().to(self.device)

            if self.cfg["obs_rms"]:
                if self.obs_rms is None:
                    self.obs_rms = RunningMeanStd(
                        shape=obs.shape[1], device=self.device
                    )
                self.obs_rms.update(obs.detach())
                obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)

            privileged_obs = self.env.get_privileged_observations().to(self.device)

            if self.cfg["privileged_obs_rms"]:
                if self.privileged_obs_rms is None:
                    self.privileged_obs_rms = RunningMeanStd(
                        shape=privileged_obs.shape[1], device=self.device
                    )
                self.privileged_obs_rms.update(privileged_obs.detach())
                privileged_obs = (
                    privileged_obs - self.privileged_obs_rms.mean
                ) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

            true_vel = self.env.get_true_vel().to(self.device)

            if self.cfg["true_vel_rms"]:
                if self.true_vel_rms is None:
                    self.true_vel_rms = RunningMeanStd(
                        shape=true_vel.shape[1], device=self.device
                    )
                self.true_vel_rms.update(true_vel.detach())
                true_vel = (true_vel - self.true_vel_rms.mean) / torch.sqrt(
                    self.true_vel_rms.var + 1e-8
                )

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    # CENet process w/ O_{t}
                    obs_history = self.env.get_observation_history()
                    if self.cfg["obs_rms"]:
                        obs_history = (obs_history - self.obs_rms.mean) / torch.sqrt(
                            self.obs_rms.var + 1e-8
                        )
                    obs_history = obs_history.reshape(self.env.num_envs, -1).to(
                        self.device
                    )  # for cenet [num_envs, 225]

                    est_next_obs, est_vel, mu, logvar, context_vec = (
                        self.cenet.before_action(obs_history, true_vel)
                    )

                    # AdaBoot
                    if self.cfg["ada_boot"]:
                        vel_input = (
                            est_vel
                            if self.env.extras["episode"]["boot_prob"].item()
                            > np.random.random()
                            else true_vel
                        )
                    else:  # Not use AdaBoot
                        vel_input = est_vel

                    # prepare observations for actor critic
                    critic_obs = torch.cat((obs, vel_input, privileged_obs), dim=-1)
                    actor_obs = torch.cat((obs, vel_input, context_vec), dim=-1)
                    obs, critic_obs, actor_obs = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        actor_obs.to(self.device),
                    )

                    # A_{t}
                    actions = self.alg.act(actor_obs, critic_obs)

                    # ============================== NEXT STEP ==============================
                    # O_{t+1}
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs = obs.to(self.device), privileged_obs.to(
                        self.device
                    )
                    true_vel = self.env.get_true_vel().to(self.device)

                    if self.cfg["obs_rms"]:
                        self.obs_rms.update(obs.detach())
                        obs = (obs - self.obs_rms.mean) / torch.sqrt(
                            self.obs_rms.var + 1e-8
                        )
                    if self.cfg["privileged_obs_rms"]:
                        self.privileged_obs_rms.update(privileged_obs.detach())
                        privileged_obs = (
                            privileged_obs - self.privileged_obs_rms.mean
                        ) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)
                    if self.cfg["true_vel_rms"]:
                        self.true_vel_rms.update(true_vel.detach())
                        true_vel = (true_vel - self.true_vel_rms.mean) / torch.sqrt(
                            self.true_vel_rms.var + 1e-8
                        )

                    self.cenet.after_action(obs)

                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        if "reward_cv" in infos:
                            rew_infos.append(infos["reward_cv"])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            # cenet update
            mean_total_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss = (
                self.cenet.update()
            )
            # policy update
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())

            if it % self.save_interval == 0:

                if self.cfg["obs_rms"]:
                    self.rms_dict["obs_rms"] = self.obs_rms
                if self.cfg["privileged_obs_rms"]:
                    self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms
                if self.cfg["true_vel_rms"]:
                    self.rms_dict["true_vel_rms"] = self.true_vel_rms

                self.save(
                    os.path.join(self.log_dir, "model_{}.pt".format(it)), infos=infos
                )

            ep_infos.clear()
            rew_infos.clear()

        self.current_learning_iteration += num_learning_iterations

        if self.cfg["obs_rms"]:
            self.rms_dict["obs_rms"] = self.obs_rms
        if self.cfg["privileged_obs_rms"]:
            self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms
        if self.cfg["true_vel_rms"]:
            self.rms_dict["true_vel_rms"] = self.true_vel_rms

        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            ),
            infos=infos,
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        info_types = [("ep_infos", "Episode"), ("rew_infos", "CV")]

        for info_type, prefix in info_types:
            if locs[info_type]:
                for key in locs[info_type][0]:  # one agent
                    infotensor = torch.tensor([], device=self.device)
                    for info in locs[info_type]:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(info[key], torch.Tensor):
                            info[key] = torch.Tensor([info[key]])
                        if len(info[key].shape) == 0:
                            info[key] = info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, info[key].to(self.device)))

                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f"{prefix}/{key}", value, locs["it"])

                    if prefix == "Episode":
                        ep_string += (
                            f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                        )

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        # if self.cfg["obs_rms"]:
        #     self.writer.add_scalar('RMS/obs_mean', self.obs_rms.mean, locs['it'])
        #     self.writer.add_scalar('RMS/obs_var', self.obs_rms.var, locs['it'])
        # if self.cfg["privileged_obs_rms"]:
        #     self.writer.add_scalar('RMS/privileged_obs_mean', self.obs_rms.mean, locs['it'])
        #     self.writer.add_scalar('RMS/privileged_obs_var', self.obs_rms.var, locs['it'])
        # if self.cfg["true_vel_rms"]:
        #     self.writer.add_scalar('RMS/true_vel_mean', self.obs_rms.mean, locs['it'])
        #     self.writer.add_scalar('RMS/true_vel_var', self.obs_rms.var, locs['it'])

        self.writer.add_scalar("CENet/beta", self.cenet.beta, locs["it"])
        self.writer.add_scalar(
            "CENet/learning_rate",
            self.cenet.optimizer.param_groups[0]["lr"],
            locs["it"],
        )
        self.writer.add_scalar("CENet/kl_loss", locs["mean_kl_loss"], locs["it"])
        self.writer.add_scalar("CENet/recon_loss", locs["mean_recon_loss"], locs["it"])
        self.writer.add_scalar("CENet/vel_loss", locs["mean_vel_loss"], locs["it"])
        self.writer.add_scalar("CENet/total_loss", locs["mean_total_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'CENet KL loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
                f"""{'CENet reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                f"""{'CENet velocity estimation loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                f"""{'CENet total loss:':>{pad}} {locs['mean_total_loss']:.4f}\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'CENet KL loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
                f"""{'CENet reconstruction loss:':>{pad}} {locs['mean_recon_loss']:.4f}\n"""
                f"""{'CENet velocity estimation loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                f"""{'CENet total loss:':>{pad}} {locs['mean_total_loss']:.4f}\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "cenet_state_dict": self.cenet.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "cenet_optimizer_state_dict": self.cenet.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
                "rms": self.rms_dict,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.cenet.load_state_dict(loaded_dict["cenet_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.cenet.optimizer.load_state_dict(
                loaded_dict["cenet_optimizer_state_dict"]
            )
        self.current_learning_iteration = loaded_dict["iter"]
        if (
            self.cfg["obs_rms"]
            or self.cfg["privileged_obs_rms"]
            or self.cfg["true_vel_rms"]
        ):
            self.rms_info = loaded_dict["rms"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_rms(self):
        return (
            self.rms_info
            if (
                self.cfg["obs_rms"]
                or self.cfg["privileged_obs_rms"]
                or self.cfg["true_vel_rms"]
            )
            else None
        )

    def get_inference_cenet(self, device=None):
        self.cenet.test_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.cenet.encoder.to(device)
        return self.cenet


class OnPolicyRunnerEst:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.vae_cfg = train_cfg["vae"]
        self.device = device
        self.env = env  # LeggedRobot

        env_cfg = self.env.cfg.env  # A1RoughEstCfg.env

        num_critic_obs = (
            env_cfg.num_observations + env_cfg.num_estvel + env_cfg.num_privileged_obs
        )
        num_actor_obs = env_cfg.num_observations + env_cfg.num_estvel

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            num_actor_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        vae_class = eval(self.cfg["vae_class_name"])  # EstNet
        self.estnet: EstNet = vae_class(device=self.device, **self.vae_cfg).to(
            self.device
        )

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        self.estnet.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [env_cfg.len_obs_history * env_cfg.num_observations],
            [env_cfg.num_estvel],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.rms_dict = {}
        if self.cfg["rms"]:  # Initialize later
            self.obs_rms = None
            self.privileged_obs_rms = None
            if self.cfg["true_vel_rms"]:
                self.true_vel_rms = None

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.estnet.train_mode()

        ep_infos = []
        rew_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # O_{t}
            obs = self.env.get_observations().to(self.device)
            if self.obs_rms is None:
                self.obs_rms = RunningMeanStd(shape=obs.shape[1], device=self.device)
            self.obs_rms.update(obs.detach())
            obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)

            privileged_obs = self.env.get_privileged_observations().to(self.device)
            if self.privileged_obs_rms is None:
                self.privileged_obs_rms = RunningMeanStd(
                    shape=privileged_obs.shape[1], device=self.device
                )
            self.privileged_obs_rms.update(privileged_obs.detach())
            privileged_obs = (
                privileged_obs - self.privileged_obs_rms.mean
            ) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

            true_vel = self.env.get_true_vel().to(self.device)
            if self.cfg["true_vel_rms"]:
                if self.true_vel_rms is None:
                    self.true_vel_rms = RunningMeanStd(
                        shape=true_vel.shape[1], device=self.device
                    )
                self.true_vel_rms.update(true_vel.detach())
                true_vel = (true_vel - self.true_vel_rms.mean) / torch.sqrt(
                    self.true_vel_rms.var + 1e-8
                )

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    # EstNet process w/ O_{t}
                    obs_history = self.env.get_observation_history()
                    obs_history = (obs_history - self.obs_rms.mean) / torch.sqrt(
                        self.obs_rms.var + 1e-8
                    )
                    obs_history = obs_history.reshape(self.env.num_envs, -1).to(
                        self.device
                    )  # for estnet [num_envs, 225]

                    est_vel = self.estnet.before_action(obs_history, true_vel)

                    # AdaBoot
                    if self.cfg["ada_boot"]:
                        vel_input = (
                            est_vel
                            if self.env.extras["episode"]["boot_prob"].item()
                            > np.random.random()
                            else true_vel
                        )
                    else:  # Not use AdaBoot
                        vel_input = est_vel

                    # prepare observations for actor critic
                    critic_obs = torch.cat((obs, vel_input, privileged_obs), dim=-1)
                    actor_obs = torch.cat((obs, vel_input), dim=-1)

                    obs, critic_obs, actor_obs = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        actor_obs.to(self.device),
                    )

                    # A_{t}
                    actions = self.alg.act(actor_obs, critic_obs)

                    # ============================== NEXT STEP ==============================
                    # O_{t+1}
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs = obs.to(self.device), privileged_obs.to(
                        self.device
                    )
                    self.obs_rms.update(obs.detach())
                    obs = (obs - self.obs_rms.mean) / torch.sqrt(
                        self.obs_rms.var + 1e-8
                    )
                    self.privileged_obs_rms.update(privileged_obs.detach())
                    privileged_obs = (
                        privileged_obs - self.privileged_obs_rms.mean
                    ) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        if "reward_cv" in infos:
                            rew_infos.append(infos["reward_cv"])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            # estnet update
            mean_vel_loss = self.estnet.update()
            # policy update
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:

                if self.cfg["obs_rms"]:
                    self.rms_dict["obs_rms"] = self.obs_rms
                if self.cfg["privileged_obs_rms"]:
                    self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms
                if self.cfg["true_vel_rms"]:
                    self.rms_dict["true_vel_rms"] = self.true_vel_rms

                self.save(
                    os.path.join(self.log_dir, "model_{}.pt".format(it)), infos=infos
                )

            ep_infos.clear()
            rew_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        infos = [self.obs_rms, self.privileged_obs_rms] if self.cfg["rms"] else None
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            ),
            infos=infos,
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        info_types = [("ep_infos", "Episode"), ("rew_infos", "CV")]

        for info_type, prefix in info_types:
            if locs[info_type]:
                for key in locs[info_type][0]:  # one agent
                    infotensor = torch.tensor([], device=self.device)
                    for info in locs[info_type]:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(info[key], torch.Tensor):
                            info[key] = torch.Tensor([info[key]])
                        if len(info[key].shape) == 0:
                            info[key] = info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, info[key].to(self.device)))

                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f"{prefix}/{key}", value, locs["it"])

                    if prefix == "Episode":
                        ep_string += (
                            f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                        )

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "EstNet/learning_rate",
            self.estnet.optimizer.param_groups[0]["lr"],
            locs["it"],
        )
        self.writer.add_scalar("EstNet/vel_loss", locs["mean_vel_loss"], locs["it"])
        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'EstNet velocity estimation loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )

        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'EstNet velocity estimation loss:':>{pad}} {locs['mean_vel_loss']:.4f}\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "estnet_state_dict": self.estnet.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "estnet_optimizer_state_dict": self.estnet.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
                "rms": self.rms_dict,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.estnet.load_state_dict(loaded_dict["estnet_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.estnet.optimizer.load_state_dict(
                loaded_dict["estnet_optimizer_state_dict"]
            )
        self.current_learning_iteration = loaded_dict["iter"]
        if (
            self.cfg["obs_rms"]
            or self.cfg["privileged_obs_rms"]
            or self.cfg["true_vel_rms"]
        ):
            self.rms_info = loaded_dict["rms"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_rms(self):
        return (
            self.rms_info
            if (
                self.cfg["obs_rms"]
                or self.cfg["privileged_obs_rms"]
                or self.cfg["true_vel_rms"]
            )
            else None
        )

    def get_inference_estnet(self, device=None):
        self.estnet.test_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.estnet.estimator.to(device)
        return self.estnet





class OnPolicyRunnerCTS:
    # Concurrent Teacher-Student (CTS) runner.
    #
    # Environment split (by index):
    #   [0 .. num_teacher_envs-1]        -> Teacher group
    #   [num_teacher_envs .. num_envs-1] -> Student group
    #
    # Teacher group:
    #   Input  : s_t (privileged state, e.g. terrain height map)
    #   Encoder: Privileged Encoder E_theta^t  [512, 256] -> z_t^t
    #   Updated: PPO reward gradient (directly from environment reward)
    #
    # Student group:
    #   Input  : o_{t-H:t} (proprioceptive obs history, NO velocity)
    #   Encoder: Proprioceptive Encoder E_theta^s  [512, 256] -> z_t^s
    #   Updated: MSE distillation  loss(z_t^s, z_t^t.detach())
    #
    # Shared policy (both groups):
    #   Actor  : pi_theta  cat(o_t, z_t)  ->  [512, 256, 128]  -> a_t
    #   Critic : V_phi     cat(s_t, z_t)  ->  [512, 256, 128]  -> V_t

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        # CTS-specific config (teacher/student split, encoder dims)
        self.cts_cfg = train_cfg.get("cts", {})
        self.device = device
        self.env = env

        env_cfg = self.env.cfg.env

        # ---- Network dimensions (Table I in the CTS paper) ----
        # Actor  : cat(o_t, z_t)  ->  num_observations + num_context
        # Critic : cat(s_t, z_t)  ->  (num_observations + num_estvel + num_privileged_obs) + num_context
        #   s_t = cat(o_t, v_t, privileged_extras)
        #        = obs(45) + lin_vel(3) + priv(190) = 238d
        #   Critic = s_t(238) + z_t(16) = 254d
        # Note: lin_vel (v_t) is removed from obs_buf by the WAQ env (cenet=True),
        #       so it must be fetched separately via env.get_true_vel().
        num_actor_obs = env_cfg.num_observations + env_cfg.num_context
        num_critic_obs = (
            env_cfg.num_observations
            + env_cfg.num_estvel
            + env_cfg.num_privileged_obs
            + env_cfg.num_context
        )

        # ---- CTS: Teacher / Student environment split ----
        self.teacher_ratio = self.cts_cfg.get("teacher_ratio", 0.5)
        self.num_teacher_envs = int(self.env.num_envs * self.teacher_ratio)
        self.num_student_envs = self.env.num_envs - self.num_teacher_envs
        print(
            f"[CTS] {self.num_teacher_envs} teacher envs + "
            f"{self.num_student_envs} student envs "
            f"(ratio={self.teacher_ratio:.2f}, total={self.env.num_envs})"
        )

        # ---- Privileged Encoder E_theta^t: s_t -> z_t^t ----
        # Input : privileged state s_t  (num_privileged_obs dims)
        # Output: context vector         (num_context dims)
        # Updated via PPO reward gradient
        priv_enc_hidden = self.cts_cfg.get("priv_enc_hidden_dims", [512, 256])
        priv_enc_layers: list[nn.Module] = []
        in_dim = env_cfg.num_privileged_obs
        for h in priv_enc_hidden:
            priv_enc_layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        priv_enc_layers.append(nn.Linear(in_dim, env_cfg.num_context))
        self.privileged_encoder = nn.Sequential(*priv_enc_layers).to(self.device)
        self.priv_enc_optimizer = optim.Adam(
            self.privileged_encoder.parameters(),
            lr=self.cts_cfg.get("priv_enc_lr", 1e-3),
        )
        print(f"[CTS] Privileged Encoder E_theta^t : {self.privileged_encoder}")

        # ---- Proprioceptive Encoder E_theta^s: o_{t-H:t} -> z_t^s ----
        # Input : flattened obs history  (len_obs_history * num_observations dims)
        # Output: context vector          (num_context dims)
        # Updated via MSE distillation (z_t^s -> z_t^t)
        student_enc_hidden = self.cts_cfg.get("student_enc_hidden_dims", [512, 256])
        student_enc_layers: list[nn.Module] = []
        in_dim = env_cfg.len_obs_history * env_cfg.num_observations
        for h in student_enc_hidden:
            student_enc_layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        student_enc_layers.append(nn.Linear(in_dim, env_cfg.num_context))
        self.student_encoder = nn.Sequential(*student_enc_layers).to(self.device)
        self.student_enc_optimizer = optim.Adam(
            self.student_encoder.parameters(),
            lr=self.cts_cfg.get("student_enc_lr", 1e-3),
        )
        print(f"[CTS] Proprioceptive Encoder E_theta^s: {self.student_encoder}")

        # Weight for MSE distillation loss
        self.mse_loss_weight = self.cts_cfg.get("mse_loss_weight", 1.0)

        # ---- Shared Actor-Critic and PPO ----
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCritic = actor_critic_class(
            num_actor_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # PPO rollout storage: all envs share the same policy
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # ---- Per-step rollout buffers for CTS update passes ----
        # Teacher: (obs, priv_obs) needed to recompute actor_obs with grad
        self._teacher_obs_buf: list[torch.Tensor] = []
        self._teacher_priv_obs_buf: list[torch.Tensor] = []
        # Student: (obs_history, priv_obs) needed for MSE distillation
        self._student_obs_hist_buf: list[torch.Tensor] = []
        self._student_priv_obs_buf: list[torch.Tensor] = []

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.rms_dict = {}
        if self.cfg["obs_rms"]:
            self.obs_rms = None
        if self.cfg["privileged_obs_rms"]:
            self.privileged_obs_rms = None

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.alg.actor_critic.train()
        self.privileged_encoder.train()
        self.student_encoder.train()

        ep_infos = []
        rew_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # o_t  (no velocity: angular vel, gravity vec, joint pos/vel, cmd vel, prev actions)
            obs = self.env.get_observations().to(self.device)
            if self.cfg["obs_rms"]:
                if self.obs_rms is None:
                    self.obs_rms = RunningMeanStd(shape=obs.shape[1], device=self.device)
                self.obs_rms.update(obs.detach())
                obs = (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + 1e-8)

            # v_t — true linear velocity (privileged; stripped from obs_buf by the WAQ env).
            # Used only by the critic so that it can observe the full state s_t.
            true_vel = self.env.get_true_vel().to(self.device)

            # s_t extras (terrain heights, disturbance force — the other privileged info)
            privileged_obs = self.env.get_privileged_observations().to(self.device)
            if self.cfg["privileged_obs_rms"]:
                if self.privileged_obs_rms is None:
                    self.privileged_obs_rms = RunningMeanStd(
                        shape=privileged_obs.shape[1], device=self.device
                    )
                self.privileged_obs_rms.update(privileged_obs.detach())
                privileged_obs = (
                    privileged_obs - self.privileged_obs_rms.mean
                ) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    # --- CTS: Context vector from teacher or student encoder ---
                    # Teacher group  [0 .. N_T-1]:
                    #   E_theta^t(s_t) -> z_t^t    (privileged state -> context)
                    # Student group  [N_T .. N-1]:
                    #   E_theta^s(o_{t-H:t}) -> z_t^s  (obs history -> context, NO velocity)

                    obs_history = self.env.get_observation_history()
                    if self.cfg["obs_rms"]:
                        obs_history = (obs_history - self.obs_rms.mean) / torch.sqrt(
                            self.obs_rms.var + 1e-8
                        )
                    obs_history = obs_history.reshape(self.env.num_envs, -1).to(
                        self.device
                    )  # [num_envs, len_H * num_obs]

                    # Teacher: privileged encoder
                    z_teacher = self.privileged_encoder(
                        privileged_obs[: self.num_teacher_envs]
                    )  # [N_T, num_context]
                    # L2-normalize onto unit hypersphere (paper Sec. II-B)
                    z_teacher = F.normalize(z_teacher, p=2, dim=-1)

                    # Student: proprioceptive encoder (obs history only, NO velocity)
                    obs_history_student = obs_history[self.num_teacher_envs :]
                    z_student = self.student_encoder(
                        obs_history_student
                    )  # [N_S, num_context]
                    # L2-normalize onto unit hypersphere (paper Sec. II-B)
                    z_student = F.normalize(z_student, p=2, dim=-1)

                    # Merge context vectors by env index
                    context_vec = torch.cat(
                        [z_teacher, z_student], dim=0
                    )  # [N, num_context]

                    # Store per-step buffers for update phase
                    self._teacher_obs_buf.append(obs[: self.num_teacher_envs].clone())
                    self._teacher_priv_obs_buf.append(
                        privileged_obs[: self.num_teacher_envs].clone()
                    )
                    self._student_obs_hist_buf.append(obs_history_student.clone())
                    self._student_priv_obs_buf.append(
                        privileged_obs[self.num_teacher_envs :].clone()
                    )

                    # Actor input:   cat(o_t, z_t)   [Table I -- Policy Network]
                    # Critic input:  cat(s_t, z_t)   [Table I -- Critic]
                    #   s_t = cat(o_t, v_t, priv_extras)  (full state per paper §II-A)
                    actor_obs = torch.cat([obs, context_vec], dim=-1)
                    critic_obs = torch.cat([obs, true_vel, privileged_obs, context_vec], dim=-1)
                    actor_obs = actor_obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)

                    # A_t
                    actions = self.alg.act(actor_obs, critic_obs)

                    # ============================== NEXT STEP ==============================
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs = obs.to(self.device), privileged_obs.to(
                        self.device
                    )
                    # Update true_vel for the next step's critic obs
                    true_vel = self.env.get_true_vel().to(self.device)

                    if self.cfg["obs_rms"]:
                        self.obs_rms.update(obs.detach())
                        obs = (obs - self.obs_rms.mean) / torch.sqrt(
                            self.obs_rms.var + 1e-8
                        )
                    if self.cfg["privileged_obs_rms"]:
                        self.privileged_obs_rms.update(privileged_obs.detach())
                        privileged_obs = (
                            privileged_obs - self.privileged_obs_rms.mean
                        ) / torch.sqrt(self.privileged_obs_rms.var + 1e-8)

                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        if "reward_cv" in infos:
                            rew_infos.append(infos["reward_cv"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Bootstrap value
                start = stop
                self.alg.compute_returns(critic_obs)

            # ---- CTS Update ----
            # 1. Teacher encoder: PPO surrogate gradient (reward signal)
            #    Must run BEFORE alg.update() which clears the storage.
            mean_priv_enc_loss = self._update_teacher_encoder()

            # 2. Student encoder: MSE distillation (student context -> teacher context)
            mean_mse_loss = self._cts_mse_distillation()

            # 3. Policy + Critic: standard PPO
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())

            if it % self.save_interval == 0:
                if self.cfg["obs_rms"]:
                    self.rms_dict["obs_rms"] = self.obs_rms
                if self.cfg["privileged_obs_rms"]:
                    self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms

                self.save(
                    os.path.join(self.log_dir, "model_{}.pt".format(it)), infos=infos
                )

            ep_infos.clear()
            rew_infos.clear()

        self.current_learning_iteration += num_learning_iterations

        if self.cfg["obs_rms"]:
            self.rms_dict["obs_rms"] = self.obs_rms
        if self.cfg["privileged_obs_rms"]:
            self.rms_dict["privileged_obs_rms"] = self.privileged_obs_rms

        self.save(
            os.path.join(
                self.log_dir,
                "model_{}.pt".format(self.current_learning_iteration),
            ),
            infos=infos,
        )

    # ------------------------------------------------------------------
    # CTS helper: update privileged encoder via PPO reward gradient
    # ------------------------------------------------------------------
    def _update_teacher_encoder(self) -> float:
        # Update E_theta^t using the PPO surrogate loss (reward signal).
        #
        # Re-runs the policy forward pass with gradient flowing through the
        # privileged encoder so PPO can back-propagate into it.
        # Must be called BEFORE self.alg.update() (which clears storage).
        if not self._teacher_obs_buf:
            return 0.0

        # Stack per-step buffers -> [T, N_T, dim]
        teacher_obs = torch.stack(self._teacher_obs_buf, dim=0)
        teacher_priv_obs = torch.stack(self._teacher_priv_obs_buf, dim=0)
        T, N_T = teacher_obs.shape[:2]
        bs = T * N_T

        teacher_obs_flat = teacher_obs.view(bs, -1)
        teacher_priv_obs_flat = teacher_priv_obs.view(bs, -1)

        # PPO storage quantities (teacher envs, no grad needed for these)
        # Use .reshape() instead of .view() because slicing [:, :N_T, :] produces
        # a non-contiguous tensor that .view() cannot handle.
        advantages_flat = (
            self.alg.storage.advantages[:, :N_T, :].detach().reshape(bs, -1)
        )
        actions_flat = self.alg.storage.actions[:, :N_T, :].detach().reshape(bs, -1)
        old_log_probs_flat = (
            self.alg.storage.actions_log_prob[:, :N_T, :].detach().reshape(bs, -1)
        )

        n_epochs = self.alg.num_learning_epochs
        mini_batch_size = max(bs // self.alg.num_mini_batches, 1)
        mean_loss = 0.0
        n_updates = 0

        self.privileged_encoder.train()

        # Freeze actor/critic weights: loss.backward() must propagate through the
        # actor network to reach z_t (since grad path is loss -> log_pi -> actor -> z_t
        # -> encoder), but we only want to UPDATE the encoder here.
        # Freezing prevents allocating .grad buffers on actor params (saves memory/compute)
        # and avoids leaving stale grads that alg.update() would then have to clear.
        for p in self.alg.actor_critic.parameters():
            p.requires_grad_(False)

        for _ in range(n_epochs):
            indices = torch.randperm(bs, device=self.device)
            for s in range(0, bs, mini_batch_size):
                idx = indices[s : s + mini_batch_size]
                if idx.numel() == 0:
                    continue

                # Recompute actor_obs with gradient through privileged encoder
                # Actor input: cat(o_t, z_t^t)  -- matches Table I
                z_teach = self.privileged_encoder(teacher_priv_obs_flat[idx])
                # L2-normalize onto unit hypersphere (paper Sec. II-B)
                z_teach = F.normalize(z_teach, p=2, dim=-1)
                actor_obs_teach = torch.cat(
                    [teacher_obs_flat[idx], z_teach], dim=-1
                )

                # Gradient path: loss -> ratio -> log_pi -> actor(frozen) -> z_teach -> encoder
                # Actor weights have requires_grad=False so no .grad allocated on them,
                # but the chain rule still passes the gradient through their activations
                # to reach the encoder via the input tensor actor_obs_teach.
                self.alg.actor_critic.act(actor_obs_teach)
                new_log_probs = self.alg.actor_critic.get_actions_log_prob(
                    actions_flat[idx]
                )

                # PPO clipped surrogate (reward gradient -> encoder)
                adv = advantages_flat[idx].squeeze(-1)
                ratio = torch.exp(
                    new_log_probs.squeeze(-1) - old_log_probs_flat[idx].squeeze(-1)
                )
                surr = -adv * ratio
                surr_clip = -adv * ratio.clamp(
                    1.0 - self.alg.clip_param, 1.0 + self.alg.clip_param
                )
                loss = torch.max(surr, surr_clip).mean()

                self.priv_enc_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.privileged_encoder.parameters(), self.alg.max_grad_norm
                )
                self.priv_enc_optimizer.step()

                mean_loss += loss.item()
                n_updates += 1

        self._teacher_obs_buf.clear()
        self._teacher_priv_obs_buf.clear()

        # Restore actor/critic gradient tracking for the upcoming alg.update() call
        for p in self.alg.actor_critic.parameters():
            p.requires_grad_(True)

        return mean_loss / max(n_updates, 1)

    # ------------------------------------------------------------------
    # CTS helper: MSE distillation  E_theta^s -> E_theta^t (supervised)
    # ------------------------------------------------------------------
    def _cts_mse_distillation(self) -> float:
        # Update E_theta^s via MSE distillation: loss(z_t^s, z_t^t.detach()).
        #
        # Paper Table III specifies separate epoch/mini-batch hyperparameters for
        # the proprioceptive encoder, independent of the PPO training configuration.
        # Configurable via cts.rec_epochs and cts.rec_num_mini_batches.
        if not self._student_obs_hist_buf:
            return 0.0

        # Stack: [T, N_S, dim]
        student_obs_hist = torch.stack(self._student_obs_hist_buf, dim=0)
        student_priv_obs = torch.stack(self._student_priv_obs_buf, dim=0)
        T, N_S = student_obs_hist.shape[:2]
        bs = T * N_S

        obs_hist_flat = student_obs_hist.view(bs, -1)
        priv_obs_flat = student_priv_obs.view(bs, -1)

        # Pre-compute frozen teacher targets once (no gradient)
        self.privileged_encoder.eval()
        with torch.no_grad():
            z_teacher_target = self.privileged_encoder(priv_obs_flat)
            # L2-normalize onto unit hypersphere (paper Sec. II-B)
            z_teacher_target = F.normalize(z_teacher_target, p=2, dim=-1)

        # Per-paper Table III: separate epoch/mini-batch loop for proprioceptive encoder
        rec_epochs = self.cts_cfg.get("rec_epochs", 4)
        rec_mini_batch_size = max(bs // self.cts_cfg.get("rec_num_mini_batches", 4), 1)

        self.student_encoder.train()
        mean_loss = 0.0
        n_updates = 0

        for _ in range(rec_epochs):
            indices = torch.randperm(bs, device=self.device)
            for s in range(0, bs, rec_mini_batch_size):
                idx = indices[s : s + rec_mini_batch_size]
                if idx.numel() == 0:
                    continue

                z_student = self.student_encoder(obs_hist_flat[idx])
                # L2-normalize onto unit hypersphere (paper Sec. II-B)
                z_student = F.normalize(z_student, p=2, dim=-1)

                loss = self.mse_loss_weight * F.mse_loss(
                    z_student, z_teacher_target[idx]
                )

                self.student_enc_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.student_encoder.parameters(), 1.0)
                self.student_enc_optimizer.step()

                mean_loss += loss.item()
                n_updates += 1

        self._student_obs_hist_buf.clear()
        self._student_priv_obs_buf.clear()

        self.privileged_encoder.train()
        return mean_loss / max(n_updates, 1)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        info_types = [("ep_infos", "Episode"), ("rew_infos", "CV")]

        for info_type, prefix in info_types:
            if locs[info_type]:
                for key in locs[info_type][0]:
                    infotensor = torch.tensor([], device=self.device)
                    for info in locs[info_type]:
                        if not isinstance(info[key], torch.Tensor):
                            info[key] = torch.Tensor([info[key]])
                        if len(info[key].shape) == 0:
                            info[key] = info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    self.writer.add_scalar(f"{prefix}/{key}", value, locs["it"])
                    if prefix == "Episode":
                        ep_string += (
                            f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                        )

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        # TensorBoard scalars
        self.writer.add_scalar(
            "CTS/mse_distillation_loss", locs["mean_mse_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "CTS/priv_enc_ppo_loss", locs["mean_priv_enc_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "CTS/priv_enc_lr",
            self.priv_enc_optimizer.param_groups[0]["lr"],
            locs["it"],
        )
        self.writer.add_scalar(
            "CTS/student_enc_lr",
            self.student_enc_optimizer.param_groups[0]["lr"],
            locs["it"],
        )
        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/learning_rate", self.alg.learning_rate, locs["it"]
        )
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward",
                statistics.mean(locs["rewbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'CTS MSE distillation loss:':>{pad}} {locs['mean_mse_loss']:.4f}\n"""
                f"""{'CTS priv-enc PPO loss:':>{pad}} {locs['mean_priv_enc_loss']:.4f}\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'CTS MSE distillation loss:':>{pad}} {locs['mean_mse_loss']:.4f}\n"""
                f"""{'CTS priv-enc PPO loss:':>{pad}} {locs['mean_priv_enc_loss']:.4f}\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "privileged_encoder_state_dict": self.privileged_encoder.state_dict(),
                "student_encoder_state_dict": self.student_encoder.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "priv_enc_optimizer_state_dict": self.priv_enc_optimizer.state_dict(),
                "student_enc_optimizer_state_dict": self.student_enc_optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
                "rms": self.rms_dict,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if "privileged_encoder_state_dict" in loaded_dict:
            self.privileged_encoder.load_state_dict(
                loaded_dict["privileged_encoder_state_dict"]
            )
        if "student_encoder_state_dict" in loaded_dict:
            self.student_encoder.load_state_dict(
                loaded_dict["student_encoder_state_dict"]
            )
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "priv_enc_optimizer_state_dict" in loaded_dict:
                self.priv_enc_optimizer.load_state_dict(
                    loaded_dict["priv_enc_optimizer_state_dict"]
                )
            if "student_enc_optimizer_state_dict" in loaded_dict:
                self.student_enc_optimizer.load_state_dict(
                    loaded_dict["student_enc_optimizer_state_dict"]
                )
        self.current_learning_iteration = loaded_dict["iter"]
        if self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"]:
            self.rms_info = loaded_dict["rms"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_rms(self):
        return (
            self.rms_info
            if (self.cfg["obs_rms"] or self.cfg["privileged_obs_rms"])
            else None
        )

    def get_inference_student_encoder(self, device=None):
        # Return the student (proprioceptive) encoder for deployment.
        self.student_encoder.eval()
        if device is not None:
            self.student_encoder.to(device)
        return self.student_encoder

    def get_inference_privileged_encoder(self, device=None):
        # Return the teacher (privileged) encoder (for analysis/logging).
        self.privileged_encoder.eval()
        if device is not None:
            self.privileged_encoder.to(device)
        return self.privileged_encoder

