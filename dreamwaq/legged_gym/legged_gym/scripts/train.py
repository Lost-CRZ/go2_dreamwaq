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
# https://github.com/curieuxjy

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, helpers
import torch

import yaml
import wandb


def save_config(env_cfg, train_cfg, config_dir):
    env_cfg_dict = dict(helpers.class_to_dict(env_cfg))
    train_cfg_dict = dict(helpers.class_to_dict(train_cfg))
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    with open(config_dir + "/env_cfg.yaml", "w") as file:
        yaml.dump(env_cfg_dict, file, default_flow_style=False)
    with open(config_dir + "/train_cfg.yaml", "w") as file:
        yaml.dump(train_cfg_dict, file, default_flow_style=False)
    return env_cfg_dict, train_cfg_dict


def train(args, flip_visual=False):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.asset.flip_visual_attachments = flip_visual  # set before env creation
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )

    print(">>> SAVED CONFIG")
    # env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    config_dir = task_registry.config_dir
    env_cfg_dict, train_cfg_dict = save_config(env_cfg, train_cfg, config_dir)


    if WANDB:
        wandb.init(
            project="dreamWaq",
            entity="crzrizer3",
            config={**env_cfg_dict, **train_cfg_dict},
        )
        wandb.tensorboard.patch(save=False, tensorboard_x=True)
        # set run name
        wandb.run.name = (
            args.task + "_" + str(env_cfg_dict["seed"]) + "_" + \
            str(env_cfg_dict["control"]["stiffness"]["joint"]) + "_" \
            + str(env_cfg_dict["control"]["damping"]["joint"])
        )
        # wandb.run.save()

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )

    if WANDB:
        wandb.finish()


if __name__ == "__main__":
    WANDB = True
    # flip_visual_attachments: rotates every visual mesh 180° around its joint axis.
    # True  — needed for Unitree URDFs exported with y-up meshes (e.g. Go2, A1).
    # False — correct for SolidWorks-exported URDFs (e.g. D1) where frames already match.
    # NOTE: affects rendering only; has NO effect on collision geometry or physics.
    FLIP_VISUAL = False
    args = get_args()
    train(args, flip_visual=FLIP_VISUAL)
