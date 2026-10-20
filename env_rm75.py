import os
import math
import copy
import time
import sys
import numpy as np
import gymnasium as gym
# The native viewer requires GLFW on macOS. Linux training without DISPLAY
# uses EGL so importing the environment does not require an X11 session.
if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
elif not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from dm_control import mjcf

from pathlib import Path

RM75_SCENE_XML = (
    Path(__file__).resolve().parent / "assets" / "rm75" / "scene.xml"
)

class RM75P2PEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ==================== 1. Gymnasium 对外接口 ====================

    def __init__(
        self,
        render_mode=None,
        seed=0,
        target_mode: str = "random",
        target_pos: list | tuple | np.ndarray = None,
        collision_penalty: float = 75.0,
        dist_success_thresh: float = 0.05,
        require_direction: bool = False,
        near_start_probability: float = 0.0,
    ):
        super().__init__()
        if collision_penalty < 0.0:
            raise ValueError(
                "collision_penalty must be non-negative"
            )
        if not np.isfinite(dist_success_thresh) or dist_success_thresh <= 0.0:
            raise ValueError("dist_success_thresh must be finite and positive")
        self.nq = 7
        self.nu = 7
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.horizon = 1024
        self.step_i = 0
        self.require_direction = bool(require_direction)
        self.near_start_probability = float(near_start_probability)
        if not np.isfinite(self.near_start_probability) or not (
            0.0 <= self.near_start_probability <= 1.0
        ):
            raise ValueError("near_start_probability must be in [0, 1]")
        if self.near_start_probability > 0.0 and not self.require_direction:
            raise ValueError("near starts require the direction task")
        self.near_start = False
        self.near_start_requested = False
        self.arrival_prefix_steps = 0
        self.goal_direction = np.array(
            [0.0, 0.0, -1.0],
            dtype=np.float64,
        )

        self.action_scale = np.full(
            self.nu,
            0.02,
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nu,),
            dtype=np.float32,
        )

        # Point observation: q(7), qd(7), TCP(3), goal(3), collision(1).
        # 点任务21维；方向任务保留历史28维基础观测并追加3维方向差。
        # Legacy point task: q + qd + tracking error + ee + goal + collision.
        # Direction task appends target_direction - current_tool_direction.
        obs_dim = 2 * self.nq + 3 + 3 + 1
        if self.require_direction:
            obs_dim += self.nq + 3  # 追踪误差7维 + 方向向量差3维，兼容历史31维模型。
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.viewer = None
        self.substeps = 10
        self.start_pos = None
        self.viewer_failed = False
        self._viewer_builtin = False
        self.ctrl_mode = os.getenv("RM75_CTRL_MODE", "position")
        if self.ctrl_mode != "position":
            raise ValueError(
                "RM75 currently supports only position actuators; "
                f"got RM75_CTRL_MODE={self.ctrl_mode!r}"
            )
        # Keep training quiet by default. Set RM75_DEBUG=1 for reset/IK logs.
        self.debug = os.getenv("RM75_DEBUG", "0") == "1"
        self._once_logged = False

        self.prev_dist = 0.0
        self.prev_direction_angle = None
        self.direction_hold_steps = 0
        self.position_reached = False
        self.direction_reached = False
        self.angle_reached = False
        self.min_direction_angle_at_position_deg = 180.0
        self.q_target = None
        self.use_ik_init = os.getenv("RM75_USE_IK", "1") == "1"  # Enable IK initialization by default
        # Stop IK inside 5 mm so the validated target box has margin below the
        # environment's 1 cm reachability acceptance threshold.
        self.ik_tolerance = 0.005
        # Reward configuration (centralized for easier tuning)
        # # The temporary TCP has no tool geometry. Use local +Z, collinear with
        # # joint_7, so rotation around the approach direction remains free.
        self.approach_axis = os.getenv("RM75_APPROACH_AXIS", "z")
        if self.approach_axis not in {"x", "y", "z", "-x", "-y", "-z"}:
            raise ValueError(
                "RM75_APPROACH_AXIS must be x, y, z, -x, -y, or -z"
            )
        self.reward_cfg = {
            "dist_success_thresh": float(dist_success_thresh),
            "direction_success_thresh": math.radians(15.0),
            "direction_hold_steps": 5,
            "direction_gate_distance": 0.20,
            "direction_gate_width": 0.15,
            "direction_progress_scale": 6.0,
            "direction_cost_weight": 0.05 if self.require_direction else 0.0,
            "time_penalty": 0.01,
            "timeout_penalty": 10.0,
            "collision_penalty": float(collision_penalty),
            "success_bonus": 100.0,
            "progress_scale": 50,  # Increased from 5.0 for stronger progress reward
            "dist_weight": 0.5 if self.require_direction else 0.20,
        }

        # Build the fixed robot/table model once. reset() only resets MjData.
        self.base_mjcf = mjcf.from_path(RM75_SCENE_XML)

        # RM75 end-effector site name (may differ per model)
        self.ee_site_name = "ee_site"  # adjust to match the model (or add a site on the terminal link)
        self._build_scene_and_model()

        self.target_mode = str(target_mode).lower().strip()
        if self.target_mode not in {"random", "fixed", "none"}:
            raise ValueError(
                "target_mode must be one of: random, fixed, none"
            )
        self.target_pos_cli = target_pos
        self.target_region = "none"
        # Fixed random box for goal sampling: [xmin, xmax, ymin, ymax, zmin, zmax] (meters)
        self.target_random_box = np.array(
            [-0.40, -0.35, -0.15, 0.15]
            if self.require_direction
            else [-0.50, -0.35, -0.30, 0.30],
            dtype=np.float64,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.arrival_prefix_steps = 0

        # Keep the compiled robot/table model and clear only per-episode
        # positions, velocities, controls, contacts, and simulation time.
        mj.mj_resetData(self.model, self.data)
        self._randomize_table_friction()

        # Define the Cartesian start before IK. Noise is applied to the IK
        # initial guess, so resets remain randomized without recompiling XML.
        self.start_pos = np.array(
            [0.4, 0.0, 0.1405 + 0.150],
            dtype=np.float64,
        )

        # Initial joint configuration
        q0 = np.array([
            -0.003093, 0.447318, 0.003770, 1.458466,
            -0.000539, 0.552761, 0.0,
        ], dtype=np.float64)
        q0 += self.rng.normal(0.0, 0.02, size=self.nq)

        # Use IK to initialize near start_pos if enabled
        if self.use_ik_init and self.start_pos is not None:
            try:
                q_ik = self._solve_ik(self.start_pos, q0)
                if q_ik is not None:
                    q0 = q_ik
                    if self.debug:
                        print("[RESET] IK solver succeeded, using IK solution for initial config")
                else:
                    if self.debug:
                        print("[RESET] IK solver failed, using default config")
            except Exception as e:
                if self.debug:
                    print(f"[RESET] IK solver error: {e}, using default config")

        self.data.qpos[:self.nq] = q0
        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)
        self.q_target = self.data.qpos[:self.nq].copy()
        # Immediately latch the current pose as the actuator reference so gravity
        # does not pull the arm down before the first env.step() call.
        self._pd_to_ctrl(self.q_target)

        # Run a short warmup to let the PD controller settle before the main loop.
        warmup_steps = max(0, int(getattr(self, "warmup_steps", 10)))
        if warmup_steps:
            self._step_sim(warmup_steps)

        self.goal_pos = None
        self._update_visualization_sites()

        # Goal sampling depending on target_mode
        mode = self.target_mode
        if mode == 'fixed':
            if self.target_pos_cli is not None:
                self.goal_pos = np.array(self.target_pos_cli, dtype=np.float64)
            else:
                self.goal_pos = np.array([-0.4, 0.0, 0.1405+0.15], dtype=np.float64)
            self.target_region = "fixed"
        elif mode == 'random':
            self.goal_pos, self.target_region = self._sample_random_goal()
        elif mode == 'none':
            self.goal_pos = None
            self.target_region = "none"

        self.near_start_requested = bool(
            self.near_start_probability > 0.0 and self.goal_pos is not None
            and self.rng.random() < self.near_start_probability
        )
        self.near_start = self.near_start_requested and self._reset_near_goal()
        self._update_visualization_sites()

        if self.debug:
            ee_after_reset = self._ee_pos()
            print(f"[RESET] use_ik_init={self.use_ik_init}")
            print("[RESET] start_pos=", self.start_pos)
            print("[RESET] goal_pos=", self.goal_pos)
            print(f"[RESET] ee_pos after init=", ee_after_reset)
            if self.start_pos is not None:
                print(f"[RESET] dist(ee, start)={np.linalg.norm(ee_after_reset - self.start_pos):.4f}")
            if (self.target_mode or '').lower() == 'random':
                print("[RESET] random_box=", self.target_random_box)

        return self._finish_reset()

    def step(self, action):
        self.step_i += 1
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.nu,):
            raise ValueError(
                f"Expected RM75 action shape {(self.nu,)}, got {action.shape}"
            )
        normalized_action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float64)
        joint_delta = self.action_scale * normalized_action

        if self.q_target is None or self.q_target.shape[0] != self.nq:
            self.q_target = self.data.qpos[:self.nq].copy()

        self.q_target = self.q_target.astype(np.float64) + joint_delta

        if self.model is not None and getattr(self.model, "njnt", 0) >= self.nq:
            jrange = self.model.jnt_range[:self.nq]
            for j in range(self.nq):
                lo, hi = jrange[j]
                if np.isfinite(lo) and np.isfinite(hi):
                    self.q_target[j] = np.clip(self.q_target[j], lo, hi)

        self._update_visualization_sites()
        self._step_sim(self.substeps)

        if self.render_mode == "human":
            self.render()

        # Collision check
        collided = self._check_collision()

        # Compute position/direction errors and the strict success flag.
        # Compute position error and point-task success.
        ee_pos = self._ee_pos()
        cfg = self.reward_cfg
        if self.goal_pos is not None:
            dist = np.linalg.norm(ee_pos - self.goal_pos)
            position_success = dist < cfg["dist_success_thresh"]
        else:
            dist = None
            position_success = False

        direction_angle = (
            self._direction_angle()
            if self.require_direction and self.goal_pos is not None
            else None
        )
        direction_success = bool(
            not self.require_direction
            or (
                direction_angle is not None
                and direction_angle < cfg["direction_success_thresh"]
            )
        )
        candidate_success = bool(position_success and direction_success and not collided)
        self.position_reached = bool(
            self.position_reached or position_success
        )
        self.direction_reached = bool(
            self.direction_reached or candidate_success
        )
        self.angle_reached = bool(self.angle_reached or direction_success)
        if position_success and direction_angle is not None:
            self.min_direction_angle_at_position_deg = min(
                self.min_direction_angle_at_position_deg,
                math.degrees(direction_angle),
            )
        if self.require_direction:
            self.direction_hold_steps = (
                self.direction_hold_steps + 1
                if candidate_success
                else 0
            )
            success = (
                self.direction_hold_steps
                >= cfg["direction_hold_steps"]
            )
        else:
            success = candidate_success

        direction_progress = self._compute_direction_progress(
            self.prev_direction_angle, direction_angle, self.prev_dist, dist,
        )

        truncated = self.step_i >= self.horizon
        timed_out = bool(truncated and not success and not collided)

        reward, reward_components = self._compute_reward(
            dist=dist,
            prev_dist=self.prev_dist,
            collided=collided,
            success=success,
            timed_out=timed_out,
            direction_progress=direction_progress,
            direction_angle=direction_angle,
        )
        self.prev_dist = dist if dist is not None else 0.0
        self.prev_direction_angle = direction_angle

        terminated = bool(success or collided)

        obs = self._get_obs()
        info = {
            "success": success,
            "collision": collided,
            "timeout": timed_out,
            "dist": dist,
            "position_success": position_success,
            "direction_success": direction_success,
            "position_reached": self.position_reached,
            "direction_reached": self.direction_reached,
            "angle_reached": self.angle_reached,
            "pose_reached": self.direction_reached,
            # # 180 degrees also represents never entering the position range.
            "min_direction_angle_at_position_deg": (
                self.min_direction_angle_at_position_deg
            ),
            "direction_angle": direction_angle,
            "direction_angle_deg": (
                math.degrees(direction_angle)
                if direction_angle is not None
                else None
            ),
            "direction_hold_steps": self.direction_hold_steps,
            "target_region": self.target_region,
            "near_start": self.near_start,
            "near_start_requested": self.near_start_requested,
            "arrival_prefix_steps": self.arrival_prefix_steps,
            "reward_components": reward_components,
            "active_table": getattr(self, "active_table", None),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            # Off-screen rendering for video recording
            if not hasattr(self, "_offscreen_renderer"):
                from mujoco import Renderer
                # HD resolution: 1280x720 (instead of 640x480)
                self._offscreen_renderer = Renderer(self.model, height=480, width=1280)

            self._update_visualization_sites()

            # Forward kinematics to update visualization
            mj.mj_forward(self.model, self.data)

            # Render and return RGB array
            self._offscreen_renderer.update_scene(self.data)
            return self._offscreen_renderer.render()

        elif self.render_mode == "human":
            if getattr(self, "viewer_failed", False):
                return
            if self.viewer is None:
                # 1) Try MuJoCo built-in viewer first
                try:
                    import mujoco.viewer as mjv
                    self.viewer = mjv.launch_passive(self.model, self.data)
                    self._viewer_builtin = True
                except Exception:
                    # 2) Fallback to mujoco-python-viewer
                    try:
                        import mujoco_viewer
                        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
                        self._viewer_builtin = False
                    except Exception as e:
                        if not self.viewer_failed:
                            print("Viewer unavailable (will not retry):", e)
                        self.viewer_failed = True
                        return
            # Draw a frame
            try:
                if self._viewer_builtin:
                    if hasattr(self.viewer, "is_running") and not self.viewer.is_running():
                        self.viewer_failed = True
                        return
                    self.viewer.sync()
                else:
                    self.viewer.render()
            except Exception:
                self.viewer_failed = True
                return

    def close(self):
        if self.viewer is not None:
            try:
                if hasattr(self.viewer, "close"):
                    self.viewer.close()
            except Exception:
                pass
            self.viewer = None

        # Clean up offscreen renderer if it exists
        if hasattr(self, "_offscreen_renderer"):
            try:
                self._offscreen_renderer.close()
            except Exception:
                pass

    # ==================== 2. 回合初始化与目标生成 ====================

    def _finish_reset(self):
        """只清回合记录，供正常重置和方向近处起点共同使用。"""
        self.step_i = 0
        self.direction_hold_steps = 0
        self.position_reached = False
        self.direction_reached = False
        self.angle_reached = False
        self.min_direction_angle_at_position_deg = 180.0

        # Initialize distance used for progress calculation (0 if no goal)

        ee_pos = self._ee_pos()
        if self.goal_pos is not None:
            self.prev_dist = np.linalg.norm(ee_pos - self.goal_pos)
        else:
            self.prev_dist = 0.0
        direction_angle = (
            self._direction_angle()
            if self.require_direction and self.goal_pos is not None
            else None
        )
        self.prev_direction_angle = direction_angle

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = {
            "q_start": self.data.qpos[:self.nq].copy(),
            "active_table": getattr(self, "active_table", None),
            "target_region": self.target_region,
            "near_start": self.near_start,
            "near_start_requested": self.near_start_requested,
            "arrival_prefix_steps": self.arrival_prefix_steps,
            "dist": self.prev_dist,
            "direction_angle": direction_angle,
            "direction_angle_deg": (
                math.degrees(direction_angle)
                if direction_angle is not None
                else None
            ),
        }
        return obs, info

    # 方向任务可选的IK近处起点初始化。
    def _reset_near_goal(self):
        """IK creates only a non-success start; every agent action remains PPO."""
        spec = mj.mjtState.mjSTATE_INTEGRATION
        saved = np.empty(mj.mj_stateSize(self.model, spec))
        mj.mj_getState(self.model, self.data, saved, spec)
        reference = self.q_target.copy()
        goal_direction = self.goal_direction.copy()
        for _ in range(8):
    #         # Practice position holding plus rotation, not another long reach.
            radius = self.rng.uniform(0.045, 0.055)
            height = self.rng.uniform(0.03, 0.04)
            azimuth = self.rng.uniform(-math.pi / 2, math.pi / 2)
            horizontal = math.sqrt(radius ** 2 - height ** 2)
            start = self.goal_pos + np.array([
                horizontal * math.cos(azimuth),
                horizontal * math.sin(azimuth), height,
            ])
            tilt = self.rng.uniform(math.radians(29), math.radians(31))
            yaw = self.rng.uniform(0.0, 2.0 * math.pi)
            self.goal_direction = np.array([
                math.sin(tilt) * math.cos(yaw),
                math.sin(tilt) * math.sin(yaw), -math.cos(tilt),
            ])
            try:
                q = self.solve_direction_ik(
                    start, reference + self.rng.normal(0.0, 0.02, self.nq),
                )
            except np.linalg.LinAlgError:
                q = None
            finally:
                self.goal_direction = goal_direction
            if q is None or not np.all(np.isfinite(q)):
                continue
            mj.mj_resetData(self.model, self.data)
            self.data.qpos[:self.nq] = q
            self.q_target = q.copy()
            self._pd_to_ctrl(self.q_target)
            self._step_sim(max(0, int(getattr(self, "warmup_steps", 10))))
            distance = float(np.linalg.norm(self._ee_pos() - self.goal_pos))
            angle = math.degrees(self._direction_angle())
            if (
                0.04 <= distance <= 0.06 and 25.0 <= angle <= 35.0
                and not self._check_collision()
            ):
                self.start_pos = self._ee_pos()
                return True
        mj.mj_setState(self.model, self.data, saved, spec)
        self.q_target = reference
        mj.mj_forward(self.model, self.data)
        return False

    def _sample_random_goal(self):
        """Uniform proposals; accept only IK-verified collision-free endpoints."""
        xmin, xmax, ymin, ymax = self.target_random_box.tolist()
        x_low, x_high = min(xmin, xmax), max(xmin, xmax)
        y_low, y_high = min(ymin, ymax), max(ymin, ymax)

        for _ in range(50):
            goal = np.array([
                self.rng.uniform(x_low, x_high),
                self.rng.uniform(y_low, y_high),
                0.2905,
            ], dtype=np.float64)
            if self._is_goal_reachable(goal):
                return goal, "uniform"

        raise RuntimeError(
            "No IK-verified collision-free goal found after 50 candidates. "
            "Check the target box, IK seeds, and collision model."
        )

    def _is_goal_reachable(self, goal):
        """Find a legal endpoint witness without moving the live robot."""
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            return False

        data = self._goal_ik_data
        ranges = self.model.jnt_range[:self.nq]
        for seed in self._goal_ik_seeds:
            mj.mj_resetData(self.model, data)
            try:
                q = self._solve_ik(goal, seed, data=data)
            except np.linalg.LinAlgError:
                continue
            if q is None:
                continue
            q = np.asarray(q, dtype=np.float64)
            if q.shape != (self.nq,) or not np.all(np.isfinite(q)):
                continue
            if np.any(q < ranges[:, 0]) or np.any(q > ranges[:, 1]):
                continue

            data.qpos[:self.nq] = q
            data.qvel[:] = 0.0
            if self._check_collision(data=data):
                continue
            error = np.linalg.norm(data.site_xpos[self.ee_sid] - goal)
            if error < self.ik_tolerance:
                return True
        return False

    # ==================== 3. 观测与奖励 ====================

    def _get_obs(self):
        q = self.data.qpos[:self.nq].copy()
        qd = self.data.qvel[:self.nq].copy()
        ee = self._ee_pos()
        goal = self.goal_pos if self.goal_pos is not None else np.zeros(3)
        coll = float(self._check_collision())
        obs = np.concatenate([
            q,
            qd,
            ee.astype(np.float64),
            goal.astype(np.float64),
            np.array([coll], dtype=np.float64)
        ]).astype(np.float32)
        if self.require_direction:
            tracking_error = (
                np.zeros(self.nq, dtype=np.float64)
                if self.q_target is None
                else self.q_target - q
            )
            split = 2 * self.nq
            obs = np.concatenate([
                obs[:split],
                tracking_error,
                obs[split:],
                self._direction_error(),
            ]).astype(np.float32)
        return obs

    def _compute_reward(
        self,
        dist,
        prev_dist,
        collided: bool,
        success: bool,
        timed_out: bool = False,
        direction_progress: float = 0.0,
        direction_angle: float | None = None,
    ):
        """合并距离、时间和终止事件奖励。"""
        cfg = self.reward_cfg

        # 距离进度奖励。
        progress = 0.0
        if (
            dist is not None and prev_dist is not None
            and (self.require_direction or prev_dist != 0.0)
        ):
            progress = cfg["progress_scale"] * (prev_dist - dist)

        # 绝对距离惩罚。
        dist_penalty = 0.0
        if dist is not None:
            dist_penalty = -cfg["dist_weight"] * dist

        # 方向任务的近处角度惩罚。
        direction_cost = 0.0
        if self.require_direction and direction_angle is not None:
            direction_cost = (
                -cfg["direction_cost_weight"] * self._direction_gate(dist)
                * float(direction_angle) / math.pi
            )

        reward = 0.0
        reward += progress
        reward += dist_penalty
        reward -= cfg["time_penalty"]
        reward += direction_progress
        reward += direction_cost

        collision_reward = 0.0
        if collided:
            collision_reward = -cfg["collision_penalty"]
            reward += collision_reward

        success_reward = 0.0
        if success:
            success_reward = cfg["success_bonus"]
            reward += success_reward

        timeout_reward = 0.0
        if timed_out:
            timeout_reward = -cfg["timeout_penalty"]
            reward += timeout_reward

        components = {
            "progress": progress,
            "dist_penalty": dist_penalty,
            "time": -cfg["time_penalty"],
            "direction_progress": direction_progress,
            "direction_cost": direction_cost,
            "collision": collision_reward,
            "success": success_reward,
            "timeout": timeout_reward,
        }
        return reward, components


    # 方向奖励仅在 require_direction=True 时参与。
    def _direction_gate(self, dist):
        """Continuous ramp: zero outside 20 cm, full importance inside 5 cm."""
        if dist is None:
            return 0.0
        outer = self.reward_cfg["direction_gate_distance"]
        width = self.reward_cfg["direction_gate_width"]
        return float(np.clip((outer - float(dist)) / width, 0.0, 1.0))

    def _compute_direction_progress(
        self, prev_angle, direction_angle, prev_dist, dist,
    ):
        """Difference of gated angular error, used with position progress.

        Combined error is 50*d + 6*gate(d)*angle/pi. The gate's maximum
        distance slope is 1/0.15, so 6/0.15 < 50: distance-only retreat cannot
        yield positive TOTAL progress, even when the tool points opposite.
        Closed-loop progress sums to zero before discounting; this heuristic
        does not claim policy invariance under PPO's discount factor.
        """
        if (
            not self.require_direction
            or prev_angle is None
            or direction_angle is None
            or prev_dist is None
            or dist is None
        ):
            return 0.0
        previous_error = (
            self._direction_gate(prev_dist) * float(prev_angle) / math.pi
        )
        current_error = (
            self._direction_gate(dist) * float(direction_angle) / math.pi
        )
        return self.reward_cfg["direction_progress_scale"] * (
            previous_error - current_error
        )

    # ==================== 4. 碰撞判定 ====================

    def _check_collision(self, *, data=None):
        """Apply the same collision rules to live or IK-probe data."""
        data = self.data if data is None else data
        mj.mj_forward(self.model, data)
        for i in range(data.ncon):
            contact = data.contact[i]
            if self._contact_pair_is_collision(contact.geom1, contact.geom2):
                return True
        return False

    def _contact_pair_is_collision(self, geom1, geom2):
        """Classify table contact or contact between two RM75 geoms."""
        geom1 = int(geom1)
        geom2 = int(geom2)
        geom1_is_table = geom1 in self.table_geom_ids
        geom2_is_table = geom2 in self.table_geom_ids
        if geom1_is_table or geom2_is_table:
            return not (geom1_is_table and geom2_is_table)

        geom1_is_floor = geom1 == self.floor_geom_id
        geom2_is_floor = geom2 == self.floor_geom_id
        if geom1_is_floor or geom2_is_floor:
            other_geom = geom2 if geom1_is_floor else geom1
            return (
                other_geom in self.robot_geom_ids
                and other_geom not in self.base_geom_ids
            )

        if (
            geom1 in self.robot_geom_ids
            and geom2 in self.robot_geom_ids
        ):
            pair = frozenset((geom1, geom2))
            return pair not in self.allowed_adjacent_robot_geom_pairs
        return False

    # ==================== 5. 模型构建与场景 ====================

    def _build_scene_and_model(self):
        # Copy base_mjcf and construct the world
        scene = copy.deepcopy(self.base_mjcf)
        world = scene.worldbody

        ee_site = scene.find("site", "ee_site")
        if ee_site is None:
            raise RuntimeError(
                "RM75 model must define site 'ee_site' on link_7"
            )
        self.ee_site_name = "ee_site"


        # Fixed tables (large boxes): use half-size
        table_size = np.array([0.8 / 2, 1.2 / 2, 0.1405 / 2])  # half-extent (x,y,z) = (1.0, 1.5, 0.1)m
        table_h = 0.0
        table_xy = np.array([0.3 + 0.8 / 2, 0.1405 / 2])

        table_names = ["table_1", "table_2"]
        table_centers = [
            np.array([table_xy[0], table_xy[1], table_h + table_size[2]], dtype=np.float64),
            np.array([-table_xy[0], table_xy[1], table_h + table_size[2]], dtype=np.float64),
        ]
        for name, center in zip(table_names, table_centers):
            world.add(
                "geom",
                name=name,
                type="box",
                size=table_size,
                pos=center.tolist(),
                rgba=[0.6, 0.4, 0.3, 1.0],
                # reset(seed=...) replaces the sliding coefficient without
                # recompiling the fixed scene topology.
                friction=[0.6, 0.005, 0.0001]
            )

        self.table_names = table_names
        self.active_table = None

        # Sites to visualize start/goal positions (start: red, goal: green)
        world.add("site", name="start_site", size=[0.012], rgba=[1.0, 0.2, 0.2, 0.9], pos=[0,0,1])
        world.add("site", name="goal_site",  size=[0.012], rgba=[0.2, 1.0, 0.2, 0.9], pos=[0,0,1])
        # # A green cylinder above the goal shows the required world -Z
        # # approach direction. It is hidden for the legacy point-only task.
        world.add(
            "site",
            name="goal_direction_site",
            type="cylinder",
            size=[0.004, 0.08],
            pos=[0, 0, 1],
            quat=[0, 1, 0, 0],
            rgba=[0.1, 0.9, 0.2, 0.85],
        )

        # Compile model
        xml = scene.to_xml_string()
        assets = scene.get_assets()
        self.model = mj.MjModel.from_xml_string(xml, assets=assets)
        self.data = mj.MjData(self.model)

        expected_joints = [f"joint_{i}" for i in range(1, 8)]
        actual_joints = [
            mj.mj_id2name(
                self.model,
                mj.mjtObj.mjOBJ_JOINT,
                i,
            )
            for i in range(self.model.njnt)
        ]
        if actual_joints != expected_joints:
            raise RuntimeError(
                f"Unexpected RM75 joint order: {actual_joints}"
            )

        expected_actuators = [
            f"joint_{i}_pos"
            for i in range(1, 8)
        ]
        actual_actuators = [
            mj.mj_id2name(
                self.model,
                mj.mjtObj.mjOBJ_ACTUATOR,
                i,
            )
            for i in range(self.model.nu)
        ]
        if actual_actuators != expected_actuators:
            raise RuntimeError(
                f"Unexpected RM75 actuator order: {actual_actuators}"
            )

        actuator_joints = [
            mj.mj_id2name(
                self.model,
                mj.mjtObj.mjOBJ_JOINT,
                int(joint_id),
            )
            for joint_id in self.model.actuator_trnid[:, 0]
        ]
        if actuator_joints != expected_joints:
            raise RuntimeError(
                "RM75 actuator-to-joint mapping is incorrect: "
                f"{actuator_joints}"
            )

        # name → id caches
        self.geom_name2id = {mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i): i
                             for i in range(self.model.ngeom)}
        self.site_name2id = {mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_SITE, i): i
                             for i in range(self.model.nsite)}
        self.ee_sid   = self.site_name2id.get(self.ee_site_name, None)
        self.start_sid = self.site_name2id["start_site"]
        self.goal_sid  = self.site_name2id["goal_site"]
        self.goal_direction_sid = self.site_name2id["goal_direction_site"]
        self.model.site_rgba[self.goal_direction_sid, 3] = (
            0.85 if self.require_direction else 0.0
        )

        self.table_geom_ids = {
            self.geom_name2id[name]
            for name in self.table_names
        }
        self.floor_geom_id = self.geom_name2id["floor"]
        self.robot_geom_ids = self._find_robot_geom_ids()
        if len(self.robot_geom_ids) != 8:
            raise RuntimeError(
                "Expected eight RM75 collision geoms "
                f"(base_link plus link_1..link_7), got "
                f"{sorted(self.robot_geom_ids)}"
            )

        # Goal IK checks must not alter the live episode state.
        self._goal_ik_data = mj.MjData(self.model)
        seed = np.array([
            -0.003093, 0.447318, 0.003770, 1.458466,
            -0.000539, 0.552761, 0.0,
        ], dtype=np.float64)
        self._goal_ik_seeds = np.tile(seed, (3, 1))
        self._goal_ik_seeds[1, 0] = 2.5
        self._goal_ik_seeds[2, 0] = -2.5
        ranges = self.model.jnt_range[:self.nq]
        self._goal_ik_seeds = np.clip(
            self._goal_ik_seeds, ranges[:, 0], ranges[:, 1]
        )

        if self.debug and not self._once_logged:
            print(f"[ENV] ee_site_name='{self.ee_site_name}', ee_sid={self.ee_sid}")
            print(f"[ENV] joints={self.nq}, actuators={self.nu}, ctrl_mode={self.ctrl_mode}")
            try:
                bid = self.model.site_bodyid[self.ee_sid] if self.ee_sid is not None else -1
                bname = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, int(bid)) if bid >= 0 else "None"
                loc = self.model.site_pos[self.ee_sid] if self.ee_sid is not None else None
                print(f"[ENV] ee_site body='{bname}' (id={bid}), local pos={loc}")
            except Exception:
                pass
            self._once_logged = True

    def _find_robot_geom_ids(self):
        """Return collision geom IDs for the base and seven RM75 links."""
        link_body_ids = {
            f"link_{index}": mj.mj_name2id(
                self.model,
                mj.mjtObj.mjOBJ_BODY,
                f"link_{index}",
            )
            for index in range(1, 8)
        }
        if -1 in link_body_ids.values():
            raise RuntimeError("RM75 link body lookup failed")

        link_geom_ids = {
            link_name: {
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) == body_id
            }
            for link_name, body_id in link_body_ids.items()
        }
        robot_geom_ids = set().union(*link_geom_ids.values())

        base_mesh_id = mj.mj_name2id(
            self.model,
            mj.mjtObj.mjOBJ_MESH,
            "base_link",
        )
        if base_mesh_id < 0:
            raise RuntimeError("RM75 base_link mesh lookup failed")
        base_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (
                int(self.model.geom_type[geom_id])
                == int(mj.mjtGeom.mjGEOM_MESH)
                and int(self.model.geom_dataid[geom_id]) == base_mesh_id
            )
        }
        self.base_geom_ids = base_geom_ids
        robot_geom_ids.update(base_geom_ids)

        # Neighboring link meshes meet at their joint by construction. These
        # contacts are structural, not robot self-collisions that should end an
        # episode. Non-neighboring RM75 geom contacts remain collisions.
        adjacent_groups = [
            base_geom_ids,
            *[
                link_geom_ids[f"link_{index}"]
                for index in range(1, 8)
            ],
        ]
        self.allowed_adjacent_robot_geom_pairs = {
            frozenset((left_geom, right_geom))
            for left_group, right_group in zip(
                adjacent_groups,
                adjacent_groups[1:],
            )
            for left_geom in left_group
            for right_geom in right_group
        }
        return robot_geom_ids

    def _randomize_table_friction(self):
        """Apply reset-seeded table friction without rebuilding the model."""
        table_friction = np.empty((len(self.table_names), 3), dtype=np.float64)
        for index, name in enumerate(self.table_names):
            table_friction[index] = [
                self.rng.uniform(0.4, 0.9),
                0.005,
                0.0001,
            ]
            geom_id = self.geom_name2id[name]
            self.model.geom_friction[geom_id] = table_friction[index]
        self.table_friction = table_friction

    # ==================== 6. 运动学与仿真控制 ====================

    def _ee_pos(self):
        if self.ee_sid is None:
            return np.zeros(3)
        mj.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.ee_sid].copy()

    # 工具方向单位向量、方向误差与夹角。
    def _ee_approach_direction(self):
        """Return the selected end-effector local axis in world coordinates."""
        mj.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.ee_sid].reshape(3, 3)
        axis_name = self.approach_axis
        sign = -1.0 if axis_name.startswith("-") else 1.0
        axis_name = axis_name[1:] if axis_name.startswith("-") else axis_name
        axis_index = {"x": 0, "y": 1, "z": 2}[axis_name]
        direction = sign * rotation[:, axis_index]
        return direction / max(float(np.linalg.norm(direction)), 1e-12)

    def _direction_error(self):
        """Vector observation that distinguishes aligned and opposite axes."""
        return self.goal_direction - self._ee_approach_direction()

    def _direction_angle(self):
        """Unsigned angle in radians between the tool and target directions."""
        alignment = float(np.dot(
            self._ee_approach_direction(),
            self.goal_direction,
        ))
        return float(np.arccos(np.clip(alignment, -1.0, 1.0)))

    def _solve_ik(
        self,
        target_pos: np.ndarray,
        initial_guess: np.ndarray,
        *,
        data=None,
    ) -> np.ndarray | None:
        """Position-only damped least squares; use supplied data when probing."""
        data = self.data if data is None else data
        q = np.asarray(initial_guess, dtype=np.float64).copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        if (
            q.shape != (self.nq,) or target_pos.shape != (3,)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(target_pos))
        ):
            return None

        ranges = self.model.jnt_range[:self.nq]
        q = np.clip(q, ranges[:, 0], ranges[:, 1])
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        damping = 1e-3
        lamI3 = (damping ** 2) * np.eye(3)

        for _ in range(500):
            data.qpos[:self.nq] = q
            data.qvel[:] = 0.0
            mj.mj_forward(self.model, data)
            e_pos = target_pos - data.site_xpos[self.ee_sid]
            if np.linalg.norm(e_pos) < self.ik_tolerance:
                return q

            mj.mj_jacSite(self.model, data, jacp, jacr, self.ee_sid)
            Jp = jacp[:, :self.nq]
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + lamI3, e_pos)
            if not np.all(np.isfinite(dq)):
                return None
            q = np.clip(
                q + 0.2 * np.clip(dq, -0.2, 0.2),
                ranges[:, 0], ranges[:, 1],
            )
        return None

    # 工具轴旋转与方向IK。
    @staticmethod
    def _axis_alignment_rotvec(current, target):
        """Shortest rotation vector that aligns one unit axis with another."""
        current = np.asarray(current, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        cross = np.cross(current, target)
        sin_angle = float(np.linalg.norm(cross))
        cos_angle = float(np.clip(np.dot(current, target), -1.0, 1.0))
        if sin_angle > 1e-9:
            return cross * (math.atan2(sin_angle, cos_angle) / sin_angle)
        if cos_angle > 0.0:
            return np.zeros(3, dtype=np.float64)
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(current, basis))) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(current, basis)
        axis /= np.linalg.norm(axis)
        return math.pi * axis

    def solve_direction_ik(self, target_pos, initial_guess):
        """Diagnostic IK for position plus tool-axis direction, with yaw free."""
        q = np.asarray(initial_guess, dtype=np.float64).copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        joint_ranges = self.model.jnt_range[:self.nq]
        damping = 1e-3
        direction_weight = 0.20

        for _ in range(500):
            self.data.qpos[:self.nq] = q
            self.data.qvel[:] = 0.0
            mj.mj_forward(self.model, self.data)
            position_error = target_pos - self.data.site_xpos[self.ee_sid]
            current_direction = self._ee_approach_direction()
            rotation_error = self._axis_alignment_rotvec(
                current_direction,
                self.goal_direction,
            )
            direction_angle = float(np.linalg.norm(rotation_error))
            if (
                np.linalg.norm(position_error) < 0.01
                and direction_angle
                < self.reward_cfg["direction_success_thresh"]
            ):
                return None if self._check_collision() else q

            mj.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_sid)
            jacobian = np.vstack([
                jacp[:, :self.nq],
                direction_weight * jacr[:, :self.nq],
            ])
            error = np.concatenate([
                position_error,
                direction_weight * rotation_error,
            ])
            system = jacobian @ jacobian.T
            dq = jacobian.T @ np.linalg.solve(
                system + (damping ** 2) * np.eye(6),
                error,
            )
            q += 0.2 * np.clip(dq, -0.2, 0.2)
            for joint_index, (lower, upper) in enumerate(joint_ranges):
                if np.isfinite(lower) and np.isfinite(upper):
                    q[joint_index] = np.clip(q[joint_index], lower, upper)
        return None

    def _pd_to_ctrl(self, q_des):
        """Send desired joint angles to the seven position actuators."""
        self.data.ctrl[:self.nu] = q_des[:self.nu]

    def _step_sim(self, n=1, hold_target=True):
        for _ in range(n):
            if hold_target and self.q_target is not None:
                self._pd_to_ctrl(self.q_target)
            mj.mj_step(self.model, self.data)

    # ==================== 7. 可视化辅助与手动交互 ====================

    def _update_visualization_sites(self):
        """更新起点、目标点及方向任务的切入方向线。"""
        if self.start_pos is not None:
            self.model.site_pos[self.start_sid] = self.start_pos
        if self.goal_pos is None:
            return
        self.model.site_pos[self.goal_sid] = self.goal_pos
        # # The 16 cm cylinder is centered 8 cm opposite the direction vector,
        # # so it terminates at the green target point for the default world -Z
        # # approach direction.
        self.model.site_pos[self.goal_direction_sid] = (
            self.goal_pos - 0.08 * self.goal_direction
        )

    def interactive_viewer(self):
        """Open the built-in MuJoCo viewer and block until the window closes.
        Tip: Use the right panel -> Actuation/Controls sliders to move joints.
        """
        try:
            import mujoco.viewer as mjv
            # On macOS, the built-in viewer requires running under 'mjpython'
            if sys.platform == "darwin" and not os.path.basename(sys.executable).startswith("mjpython"):
                print("Tip: On macOS, run with 'mjpython env.py --interactive' for the built-in viewer.")
                return self._fallback_interactive_viewer()
        except Exception as e:
            print("Built-in viewer unavailable:", e)
            # Fallback to non-blocking path
            self.render_mode = "human"
            self.render()
            print("Fallback viewer started (non-blocking). Close the window or Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                return

        # Ensure model/data exist
        if self.model is None or self.data is None:
            self._build_scene_and_model()
            mj.mj_forward(self.model, self.data)

        print("[Interactive] Use the right panel's Actuation sliders to move joints. Close the window to exit.")
        q_hold = self.data.qpos[:self.nq].copy()
        try:
            with mjv.launch_passive(self.model, self.data) as viewer:
                while viewer.is_running():
                    self._update_visualization_sites()
                    self._pd_to_ctrl(q_hold)
                    mj.mj_step(self.model, self.data)
                    viewer.sync()
        except RuntimeError as e:
            print("Built-in viewer couldn't start:", e)
            return self._fallback_interactive_viewer()

    def _fallback_interactive_viewer(self):
        """Fallback loop using mujoco-python-viewer (GLFW)."""
        try:
            import mujoco_viewer
        except Exception as e:
            print("Fallback viewer unavailable:", e)
            print("Hint: fix ~/.config/mujoco_viewer permissions or install mujoco-python-viewer:")
            print("  pip install mujoco-python-viewer glfw")
            return
        viewer = None
        try:
            viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            print("[Interactive] Fallback viewer active. Press Ctrl+C to exit.")
            while True:
                self._update_visualization_sites()
                self._pd_to_ctrl(self.data.qpos[:self.nq])
                mj.mj_step(self.model, self.data)
                viewer.render()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print("Fallback viewer closed:", e)
        finally:
            try:
                if viewer is not None:
                    viewer.close()
            except Exception:
                pass
