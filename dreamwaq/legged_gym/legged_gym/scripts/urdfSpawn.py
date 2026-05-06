"""
urdfSpawn.py  — Generic URDF inspector for Isaac Gym
=====================================================
Spawns any URDF in Isaac Gym, prints body / joint / DOF info,
and animates through every DOF sequentially so you can visually
verify limits and mesh attachments.

Usage
-----
  python urdfSpawn.py --urdf <path/to/robot.urdf> [--asset_root <dir>] [--fixed]

  --urdf        Path to the URDF file, relative to --asset_root.
                Default: robots/go2/urdf/go2.urdf
  --asset_root  Root directory that Isaac Gym uses to resolve the URDF
                and any relative mesh paths inside it.
                Default: ../resources  (relative to this script)
  --fixed       Fix the base link (default: True).  Pass --no-fixed to free it.
  --height      Spawn height in metres (default: 0.5).

Standard Isaac Gym flags (--physics_engine, --num_threads, --use_gpu …)
are also accepted and forwarded to the simulator.

Controls
--------
  Close the viewer window to quit.
"""

import argparse
import math
import os
import sys

import numpy as np
from isaacgym import gymapi, gymutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DT = 1.0 / 60.0
ENV_LOWER = gymapi.Vec3(-2.0, -2.0, 0.0)
ENV_UPPER = gymapi.Vec3(2.0, 2.0, 4.0)

ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

# Default joint angles per robot (by DOF name).
# Used in --simulate mode to set the initial standing pose.
KNOWN_DEFAULTS = {
    # D1
    "FL_hip_joint":   0.0,
    "FR_hip_joint":   0.0,
    "RL_hip_joint":   0.0,
    "RR_hip_joint":   0.0,
    "FL_thigh_joint": 0.72,
    "FR_thigh_joint": 0.72,
    "RL_thigh_joint": 0.72,
    "RR_thigh_joint": 0.72,
    "FL_calf_joint":  -1.44,
    "FR_calf_joint":  -1.44,
    "RL_calf_joint":  -1.44,
    "RR_calf_joint":  -1.44,
    # Go2 / A1 (hip/thigh/calf same names, different values)
    "FL_calf_joint_go2":  -1.5,   # placeholder — add overrides as needed
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clamp(x, lo, hi):
    return max(min(x, hi), lo)


def print_asset_info(gym_handle, asset):
    n_bodies = gym_handle.get_asset_rigid_body_count(asset)
    n_joints = gym_handle.get_asset_joint_count(asset)
    n_dofs = gym_handle.get_asset_dof_count(asset)
    print(f"\n{'='*60}")
    print(f"  Bodies : {n_bodies}  |  Joints : {n_joints}  |  DOFs : {n_dofs}")
    print(f"{'='*60}")

    print("\nBodies:")
    for i in range(n_bodies):
        print(f"  {i:3d}: {gym_handle.get_asset_rigid_body_name(asset, i)}")

    print("\nJoints:")
    for i in range(n_joints):
        print(f"  {i:3d}: {gym_handle.get_asset_joint_name(asset, i)}")

    print("\nDOFs:")
    dof_props = gym_handle.get_asset_dof_properties(asset)
    for i in range(n_dofs):
        name = gym_handle.get_asset_dof_names(asset)[i]
        dof_type = gym_handle.get_dof_type_string(gym_handle.get_asset_dof_type(asset, i))
        lo = dof_props["lower"][i]
        hi = dof_props["upper"][i]
        limited = bool(dof_props["hasLimits"][i])
        print(f"  {i:3d}: {name:<30s}  type={dof_type:<12s}  limited={limited}"
              + (f"  [{lo:.3f}, {hi:.3f}]" if limited else ""))
    print()


def build_dof_tables(gym_handle, asset, spawn_height):
    """Return arrays needed to drive DOF animation."""
    n_dofs = gym_handle.get_asset_dof_count(asset)
    dof_props = gym_handle.get_asset_dof_properties(asset)
    dof_states = np.zeros(n_dofs, dtype=gymapi.DofState.dtype)
    dof_positions = dof_states["pos"]

    lower_limits = np.copy(dof_props["lower"])
    upper_limits = np.copy(dof_props["upper"])
    speeds = np.zeros(n_dofs)

    for i in range(n_dofs):
        dof_type = gym_handle.get_asset_dof_type(asset, i)
        if not dof_props["hasLimits"][i]:
            if dof_type == gymapi.DOF_ROTATION:
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            else:
                lower_limits[i] = -1.0
                upper_limits[i] = 1.0
        else:
            if dof_type == gymapi.DOF_ROTATION:
                lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)

        # Start at 0 (neutral/rest pose for most URDFs)
        dof_positions[i] = 0.0

        range_ = upper_limits[i] - lower_limits[i]
        if dof_type == gymapi.DOF_ROTATION:
            speeds[i] = clamp(2.0 * range_, 0.25 * math.pi, 3.0 * math.pi)
        else:
            speeds[i] = clamp(2.0 * range_, 0.1, 7.0)

    return dof_states, dof_positions, lower_limits, upper_limits, speeds


def step_animation(anim_state, current_dof, dof_positions,
                   lower_limits, upper_limits, speeds, n_dofs):
    speed = speeds[current_dof]

    if anim_state == ANIM_SEEK_LOWER:
        dof_positions[current_dof] -= speed * DT
        if dof_positions[current_dof] <= lower_limits[current_dof]:
            dof_positions[current_dof] = lower_limits[current_dof]
            anim_state = ANIM_SEEK_UPPER

    elif anim_state == ANIM_SEEK_UPPER:
        dof_positions[current_dof] += speed * DT
        if dof_positions[current_dof] >= upper_limits[current_dof]:
            dof_positions[current_dof] = upper_limits[current_dof]
            anim_state = ANIM_SEEK_DEFAULT

    if anim_state == ANIM_SEEK_DEFAULT:
        # return to neutral (0.0)
        neutral = 0.0
        if dof_positions[current_dof] > neutral:
            dof_positions[current_dof] -= speed * DT
            if dof_positions[current_dof] <= neutral:
                dof_positions[current_dof] = neutral
                anim_state = ANIM_FINISHED
        else:
            dof_positions[current_dof] += speed * DT
            if dof_positions[current_dof] >= neutral:
                dof_positions[current_dof] = neutral
                anim_state = ANIM_FINISHED

    elif anim_state == ANIM_FINISHED:
        current_dof = (current_dof + 1) % n_dofs
        anim_state = ANIM_SEEK_LOWER

    return anim_state, current_dof


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- parse our own args before handing the rest to gymutil ---------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.join(script_dir, "..", "..", "resources")

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--urdf", default="robots/d1/urdf/d1.urdf",
                            help="URDF path relative to --asset_root")
    pre_parser.add_argument("--asset_root", default=default_root,
                            help="Root directory for asset loading")
    pre_parser.add_argument("--fixed", dest="fixed", action="store_true", default=True,
                            help="Fix the base link (default)")
    pre_parser.add_argument("--no-fixed", dest="fixed", action="store_false",
                            help="Free the base link")
    pre_parser.add_argument("--height", type=float, default=0.57,
                            help="Spawn height in metres (default 0.57)")
    pre_parser.add_argument("--simulate", action="store_true", default=False,
                            help="Simulate with PD hold at default joint angles instead of animating")
    pre_parser.add_argument("--stiffness", type=float, default=100.0,
                            help="PD stiffness used in simulate mode (default 100.0)")
    pre_parser.add_argument("--damping", type=float, default=5.0,
                            help="PD damping used in simulate mode (default 5.0)")
    pre_parser.add_argument("--flip", dest="flip", action="store_true", default=False,
                            help="Set flip_visual_attachments (needed for some URDFs e.g. Go2)")
    pre_parser.add_argument("--no-flip", dest="flip", action="store_false",
                            help="Disable flip_visual_attachments (default, correct for SolidWorks exports)")
    pre_parser.add_argument("--interactive", action="store_true", default=False,
                            help="Spawn in stance pose (fixed in air) and control joints interactively with keyboard")
    pre_parser.add_argument("--delta", type=float, default=0.05,
                            help="Joint angle step per keypress in interactive mode (rad, default 0.05)")
    our_args, remaining = pre_parser.parse_known_args()

    # gymutil.parse_arguments needs sys.argv; feed it only the unrecognised args
    sys.argv = [sys.argv[0]] + remaining
    gym_args = gymutil.parse_arguments(description="URDF Spawner — Isaac Gym")

    asset_root = os.path.abspath(our_args.asset_root)
    urdf_file = our_args.urdf
    simulate_mode = our_args.simulate
    interactive_mode = our_args.interactive
    joint_delta = our_args.delta
    # In simulate mode default to free base; honour explicit --fixed if passed
    if simulate_mode and "--fixed" not in sys.argv:
        fixed_base = False
    else:
        fixed_base = our_args.fixed
    spawn_height = our_args.height
    flip_visual = our_args.flip
    pd_stiffness = our_args.stiffness
    pd_damping = our_args.damping

    print(f"\nAsset root : {asset_root}")
    print(f"URDF file  : {urdf_file}")
    if interactive_mode:
        mode_str = "interactive (stance + keyboard joint control)"
    elif simulate_mode:
        mode_str = "simulate (PD hold)"
    else:
        mode_str = "animate DOFs"
    print(f"Mode       : {mode_str}")
    print(f"Fixed base : {fixed_base}")
    print(f"Height     : {spawn_height} m")
    print(f"Flip visual: {flip_visual}\n")

    # --- Isaac Gym setup ------------------------------------------------------
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = DT
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if gym_args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = gym_args.num_threads
        sim_params.physx.use_gpu = gym_args.use_gpu

    sim_params.use_gpu_pipeline = False  # keep tensors on CPU for simplicity

    sim = gym.create_sim(
        gym_args.compute_device_id,
        gym_args.graphics_device_id,
        gym_args.physics_engine,
        sim_params,
    )
    if sim is None:
        print("ERROR: Failed to create sim")
        sys.exit(1)

    # ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # viewer
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        print("ERROR: Failed to create viewer")
        sys.exit(1)
    gym.viewer_camera_look_at(
        viewer, None,
        gymapi.Vec3(2.0, 2.0, 1.5),
        gymapi.Vec3(0.0, 0.0, 0.3),
    )

    # --- load asset -----------------------------------------------------------
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fixed_base
    asset_options.flip_visual_attachments = flip_visual

    print(f"Loading '{urdf_file}' …")
    asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
    if asset is None:
        print(f"ERROR: Could not load URDF '{urdf_file}' from '{asset_root}'")
        sys.exit(1)

    print_asset_info(gym, asset)

    # --- env + actor ----------------------------------------------------------
    env = gym.create_env(sim, ENV_LOWER, ENV_UPPER, 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, spawn_height)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor = gym.create_actor(env, asset, pose, "robot", 0, 0)

    n_dofs = gym.get_asset_dof_count(asset)
    if n_dofs == 0:
        print("No DOFs found — displaying static model. Close viewer to quit.")
        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    elif interactive_mode:
        # ---- Stance pose + interactive joint control (torque / effort mode) -
        # Mirrors training: τ = Kp*(target + default - pos) - Kd*vel
        # applied via set_dof_actuation_force_tensor, drive mode = EFFORT.
        from isaacgym import gymtorch
        import torch

        dof_names = gym.get_asset_dof_names(asset)
        dof_props = gym.get_actor_dof_properties(env, actor)
        asset_dof_props = gym.get_asset_dof_properties(asset)

        # Build stance pose from KNOWN_DEFAULTS (0.0 for unknown joints)
        dof_states = np.zeros(n_dofs, dtype=gymapi.DofState.dtype)
        default_positions = np.zeros(n_dofs, dtype=np.float32)
        for i, name in enumerate(dof_names):
            angle = KNOWN_DEFAULTS.get(name, 0.0)
            dof_states["pos"][i] = angle
            default_positions[i] = angle
        # action offsets on top of default_positions (starts at zero = stance)
        action_offsets = np.zeros(n_dofs, dtype=np.float32)

        # Clamp limits (use ±π / ±1.0 for unlimited DOFs)
        lower_limits = np.copy(asset_dof_props["lower"])
        upper_limits = np.copy(asset_dof_props["upper"])
        for i in range(n_dofs):
            if not asset_dof_props["hasLimits"][i]:
                dof_type = gym.get_asset_dof_type(asset, i)
                if dof_type == gymapi.DOF_ROTATION:
                    lower_limits[i], upper_limits[i] = -math.pi, math.pi
                else:
                    lower_limits[i], upper_limits[i] = -1.0, 1.0

        # Set drive mode to EFFORT (torque) — no internal PD, we compute it
        for i in range(n_dofs):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
            dof_props["stiffness"][i] = 0.0
            dof_props["damping"][i] = 0.0
        gym.set_actor_dof_properties(env, actor, dof_props)

        # Snap to stance pose, zero velocity
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

        # Acquire DOF state tensor (pos + vel for all DOFs in the env)
        gym.prepare_sim(sim)
        dof_state_tensor = gym.acquire_dof_state_tensor(sim)
        dof_state_torch = gymtorch.wrap_tensor(dof_state_tensor)  # (n_dofs, 2)
        root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
        root_state_torch = gymtorch.wrap_tensor(root_state_tensor)  # (1, 13): px py pz qx qy qz qw vx vy vz wx wy wz
        torques = np.zeros(n_dofs, dtype=np.float32)

        # Subscribe keyboard events
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT,    "prev_joint")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT,   "next_joint")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP,      "increase")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN,    "decrease")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_PAGE_UP, "increase_large")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_PAGE_DOWN, "decrease_large")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R,       "reset")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE,   "print_all")

        current_joint = [0]  # mutable container so inner fn can update

        def status_line():
            j = current_joint[0]
            effective = default_positions[j] + action_offsets[j]
            print(f"\r  Joint [{j:2d}/{n_dofs-1}] {dof_names[j]:<30s}"
                  f"  action_offset={action_offsets[j]:+.4f}  target={effective:+.4f} rad"
                  f"  limits=[{lower_limits[j]:.3f}, {upper_limits[j]:.3f}]   ",
                  end="", flush=True)

        print("\nInteractive mode controls:")
        print(f"  LEFT / RIGHT       : previous / next joint")
        print(f"  UP / DOWN          : +/- {joint_delta:.3f} rad action offset")
        print(f"  PAGE_UP / PAGE_DOWN: +/- {joint_delta*10:.3f} rad (10× step)")
        print(f"  R                  : reset all joints to stance defaults")
        print(f"  SPACE              : print full joint state")
        print( "  Close window       : quit\n")
        status_line()

        while not gym.query_viewer_has_closed(viewer):
            for evt in gym.query_viewer_action_events(viewer):
                if evt.value == 0:   # key-release, ignore
                    continue
                j = current_joint[0]
                if evt.action == "prev_joint":
                    current_joint[0] = (j - 1) % n_dofs
                    status_line()
                elif evt.action == "next_joint":
                    current_joint[0] = (j + 1) % n_dofs
                    status_line()
                elif evt.action in ("increase", "increase_large"):
                    step = joint_delta if evt.action == "increase" else joint_delta * 10
                    effective_target = default_positions[j] + action_offsets[j] + step
                    effective_target = clamp(effective_target, lower_limits[j], upper_limits[j])
                    action_offsets[j] = effective_target - default_positions[j]
                    status_line()
                elif evt.action in ("decrease", "decrease_large"):
                    step = joint_delta if evt.action == "decrease" else joint_delta * 10
                    effective_target = default_positions[j] + action_offsets[j] - step
                    effective_target = clamp(effective_target, lower_limits[j], upper_limits[j])
                    action_offsets[j] = effective_target - default_positions[j]
                    status_line()
                elif evt.action == "reset":
                    action_offsets[:] = 0.0
                    print("\nReset to stance defaults.")
                    status_line()
                elif evt.action == "print_all":
                    gym.refresh_actor_root_state_tensor(sim)
                    rs = root_state_torch[0].cpu().numpy()
                    print("\nBase state:")
                    print(f"  pos      : x={rs[0]:+.4f}  y={rs[1]:+.4f}  z={rs[2]:+.4f} m")
                    print(f"  quat     : x={rs[3]:+.4f}  y={rs[4]:+.4f}  z={rs[5]:+.4f}  w={rs[6]:+.4f}")
                    print(f"  lin_vel  : x={rs[7]:+.4f}  y={rs[8]:+.4f}  z={rs[9]:+.4f} m/s")
                    print(f"  ang_vel  : x={rs[10]:+.4f}  y={rs[11]:+.4f}  z={rs[12]:+.4f} rad/s")
                    gym.refresh_dof_state_tensor(sim)
                    cur_pos = dof_state_torch[:, 0].cpu().numpy()
                    cur_vel = dof_state_torch[:, 1].cpu().numpy()
                    print("\nFull joint state:")
                    for i, name in enumerate(dof_names):
                        marker = ">>>" if i == current_joint[0] else "   "
                        eff = default_positions[i] + action_offsets[i]
                        print(f"  {marker} {i:2d}: {name:<30s}  offset={action_offsets[i]:+.4f}  target={eff:+.4f}  pos={cur_pos[i]:+.4f}  vel={cur_vel[i]:+.5f} rad/s")
                    status_line()

            # --- torque control matching training ----------------------------
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)
            dof_pos = dof_state_torch[:, 0].cpu().numpy()
            dof_vel = dof_state_torch[:, 1].cpu().numpy()
            # τ = Kp * (action_offset + default_pos - pos) - Kd * vel
            torques[:] = (pd_stiffness * (action_offsets + default_positions - dof_pos)
                          - pd_damping * dof_vel)
            torque_tensor = gymtorch.unwrap_tensor(
                torch.from_numpy(torques))
            gym.set_dof_actuation_force_tensor(sim, torque_tensor)

            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Draw green axis line for the active joint
            gym.clear_lines(viewer)
            dof_handle = gym.get_actor_dof_handle(env, actor, current_joint[0])
            frame = gym.get_dof_frame(env, dof_handle)
            gymutil.draw_line(
                frame.origin,
                frame.origin + frame.axis * 0.4,
                gymapi.Vec3(0.0, 1.0, 0.0),
                gym, viewer, env,
            )

            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    elif simulate_mode:
        # ---- Natural physics simulation with PD hold at default angles ------
        dof_names = gym.get_asset_dof_names(asset)
        dof_props = gym.get_actor_dof_properties(env, actor)

        # Build initial position array from known defaults (0.0 for unknowns)
        dof_states = np.zeros(n_dofs, dtype=gymapi.DofState.dtype)
        target_positions = np.zeros(n_dofs, dtype=np.float32)
        for i, name in enumerate(dof_names):
            angle = KNOWN_DEFAULTS.get(name, 0.0)
            dof_states["pos"][i] = angle
            target_positions[i] = angle

        # Set PD drive mode and gains
        for i in range(n_dofs):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = pd_stiffness
            dof_props["damping"][i] = pd_damping
        gym.set_actor_dof_properties(env, actor, dof_props)

        # Snap to starting pose
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_POS)
        gym.set_actor_dof_position_targets(env, actor, target_positions)

        print(f"Simulating with PD hold  (stiffness={pd_stiffness}, damping={pd_damping})")
        print("Initial joint positions:")
        for i, name in enumerate(dof_names):
            print(f"  {name:<30s}: {dof_states['pos'][i]:.3f} rad")
        print("Close viewer to quit.\n")

        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    else:
        # ---- DOF animation mode ---------------------------------------------
        dof_states, dof_positions, lower_limits, upper_limits, speeds = \
            build_dof_tables(gym, asset, spawn_height)

        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_POS)

        anim_state = ANIM_SEEK_LOWER
        current_dof = 0
        dof_names = gym.get_asset_dof_names(asset)

        print(f"Animating DOF 0: {dof_names[0]}")
        prev_dof = 0

        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            anim_state, current_dof = step_animation(
                anim_state, current_dof, dof_positions,
                lower_limits, upper_limits, speeds, n_dofs,
            )

            if current_dof != prev_dof:
                print(f"Animating DOF {current_dof}: {dof_names[current_dof]}")
                prev_dof = current_dof

            gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_POS)

            # draw axis line for the active DOF
            gym.clear_lines(viewer)
            dof_handle = gym.get_actor_dof_handle(env, actor, current_dof)
            frame = gym.get_dof_frame(env, dof_handle)
            gymutil.draw_line(
                frame.origin,
                frame.origin + frame.axis * 0.4,
                gymapi.Vec3(1.0, 0.0, 0.0),
                gym, viewer, env,
            )

            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    print("Exiting.")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
