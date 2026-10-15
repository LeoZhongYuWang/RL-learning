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


def _normalize_direction(direction, *, name="direction"):
    """Return a finite unit vector and reject an undefined direction."""
    vector = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} must be a finite non-zero 3D vector")
    return vector / norm


def _quat_from_z_axis(direction):
    """Return a MuJoCo wxyz quaternion rotating local +Z to direction."""
    target = _normalize_direction(direction)
    source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    axis = np.cross(source, target)
    axis /= np.linalg.norm(axis)
    half_angle = 0.5 * math.acos(dot)
    return np.concatenate((
        np.array([math.cos(half_angle)], dtype=np.float64),
        axis * math.sin(half_angle),
    ))

class RM75P2PEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        seed=0,
        target_mode: str = "random",
        target_pos: list | tuple | np.ndarray = None,
        target_sampling: str = "uniform",
        collision_penalty: float = 75.0,
        approach_task: bool = False,
        target_direction: list | tuple | np.ndarray = (0.0, 0.0, -1.0),
        target_direction_mode: str = "fixed",
        target_tilt_min_deg: float = 0.0,
        target_tilt_max_deg: float = 15.0,
        target_azimuth_min_deg: float = -180.0,
        target_azimuth_max_deg: float = 180.0,
        approach_position_box: list | tuple | np.ndarray | None = None,
        require_pose_feasible: bool = False,
        pose_feasibility_attempts: int = 20,
        orient_success_threshold_deg: float = 15.0,
        pose_task: bool = False,
    ):
        super().__init__()
        if collision_penalty < 0.0:
            raise ValueError(
                "collision_penalty must be non-negative"
            )
        self.nq = 7
        self.nu = 7
        self.pose_task = bool(pose_task)
        self.approach_task = bool(approach_task or self.pose_task)
        self.target_direction_mode = str(
            target_direction_mode
        ).lower().strip()
        if self.target_direction_mode not in {"fixed", "random_cone"}:
            raise ValueError(
                "target_direction_mode must be fixed or random_cone"
            )
        self.target_direction = _normalize_direction(
            target_direction,
            name="target_direction",
        )
        self.fixed_target_direction = self.target_direction.copy()
        angle_values = np.asarray(
            [
                target_tilt_min_deg,
                target_tilt_max_deg,
                target_azimuth_min_deg,
                target_azimuth_max_deg,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(angle_values).all():
            raise ValueError("target direction angle bounds must be finite")
        if not 0.0 <= target_tilt_min_deg <= target_tilt_max_deg <= 90.0:
            raise ValueError(
                "target tilt bounds must satisfy 0 <= min <= max <= 90"
            )
        if target_azimuth_min_deg > target_azimuth_max_deg:
            raise ValueError(
                "target azimuth min must not exceed target azimuth max"
            )
        self.target_tilt_min_deg = float(target_tilt_min_deg)
        self.target_tilt_max_deg = float(target_tilt_max_deg)
        self.target_azimuth_min_deg = float(target_azimuth_min_deg)
        self.target_azimuth_max_deg = float(target_azimuth_max_deg)
        self.target_tilt_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(-self.target_direction[2], -1.0, 1.0)
                )
            )
        )
        self.target_azimuth_deg = float(
            np.degrees(
                np.arctan2(
                    self.target_direction[1],
                    self.target_direction[0],
                )
            )
        )
        self.require_pose_feasible = bool(require_pose_feasible)
        if int(pose_feasibility_attempts) <= 0:
            raise ValueError("pose_feasibility_attempts must be positive")
        self.pose_feasibility_attempts = int(pose_feasibility_attempts)
        if (
            not np.isfinite(orient_success_threshold_deg)
            or orient_success_threshold_deg <= 0.0
            or orient_success_threshold_deg > 180.0
        ):
            raise ValueError(
                "orient_success_threshold_deg must be within (0, 180]"
            )
        self.orient_success_threshold_deg = float(
            orient_success_threshold_deg
        )
        self.goal_direction_length = 0.12
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        # The pose task gets 12 seconds at 50 Hz.  This is long enough for the
        # full workspace move without returning to the very long legacy
        # horizon that slows target turnover during training.
        self.horizon = 600 if self.pose_task else 1024
        self.step_i = 0

        self.action_scale = np.full(
            self.nu,
            0.025 if self.pose_task else 0.02,
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nu,),
            dtype=np.float32,
        )

        if self.pose_task:
            # q(7), qd(7), relative position(3), rotation vector(3).
            obs_low = np.array(
                [-1.0] * 14 + [-2.0] * 3 + [-1.0] * 3,
                dtype=np.float32,
            )
            self.observation_space = gym.spaces.Box(
                low=obs_low,
                high=-obs_low,
                dtype=np.float32,
            )
        else:
            # Point task: q + qd + tracking error + ee + goal + collision = 28.
            # Legacy approach adds current/desired tool axes = 34.
            obs_dim = 3 * self.nq + 3 + 3 + 1
            if self.approach_task:
                obs_dim += 6
            obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                -obs_high,
                obs_high,
                dtype=np.float32,
            )

        self.viewer = None
        self.substeps = 10
        self.goal_quat = np.array([0,0,0,1], dtype=np.float64)  # orientation is fixed or optional
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
        self.prev_angle_error = None
        self._milestone_20cm_claimed = False
        self._milestone_10cm_claimed = False
        self._active_success_claimed = False
        self.ever_active_success = False
        self.q_target = None
        self.pose_keypoint_length = 0.15
        self.pose_velocity_scale = 2.5
        self.pose_position_error_scale = 0.5
        self.pose_success_hold_steps = 5
        self.pose_success_streak = 0
        self.previous_action = np.zeros(self.nu, dtype=np.float64)
        self.use_ik_init = os.getenv("RM75_USE_IK", "1") == "1"  # Enable IK initialization by default
        # Stop IK inside 5 mm so the validated target box has margin below the
        # environment's 1 cm reachability acceptance threshold.
        self.ik_tolerance = 0.005
        # Reward configuration (centralized for easier tuning)
        self.orient_axis = os.getenv("RM75_ORIENT_AXIS", "z")  # 'x','y','z','-x','-y','-z'
        self.reward_cfg = {
            "dist_success_thresh": 0.05,
            "time_penalty": 0.01,
            "timeout_penalty": 10.0,
            "collision_penalty": float(collision_penalty),
            "success_bonus": 100.0,
            # Curriculum success is a measurement, not a second task goal.
            # A large bonus here made Stage 3 (10 cm / 35 deg) more attractive
            # than continuing toward the strict 5 cm / final-angle target.
            "active_success_bonus": 0.0,
            "orient_weight": 0.0,
            "orient_power": 2.0,
            "orient_success_thresh": np.deg2rad(
                self.orient_success_threshold_deg
            ),
            "orient_progress_scale": 10.0,
            "orient_error_weight": 0.05,
            "orient_gate_start_dist": 0.25,
            "pose_progress_scale": 5.0,
            "pose_error_weight": 0.05,
            "progress_scale": 50,  # Increased from 5.0 for stronger progress reward
            "dist_weight": 0.03,
            "milestone_20cm_bonus": 5.0,
            "milestone_10cm_bonus": 15.0,
            # Keep the legacy point-reaching milestones above unchanged.  The
            # approach task uses smaller hints so they cannot dominate the
            # dense pose potential or the strict success bonus.
            "approach_milestone_20cm_bonus": 2.0,
            "approach_precision_milestone_bonus": 5.0,
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
        self.target_sampling = str(target_sampling).lower().strip()
        if self.target_sampling not in {"uniform", "hard_mixture"}:
            raise ValueError(
                "target_sampling must be one of: uniform, hard_mixture"
            )
        self.target_region = "none"
        # Fixed random box for goal sampling: [xmin, xmax, ymin, ymax, zmin, zmax] (meters)
        self.target_random_box = np.array([
            -0.5, -0.35,   # X range
            -0.3, 0.3   ,   # Y range
            # 0.18, 0.45   # Z range (approx table top + margin)
        ], dtype=np.float64)
        if approach_position_box is None:
            self.approach_position_box = np.array(
                [-0.45, -0.35, -0.18, 0.18],
                dtype=np.float64,
            )
        else:
            self.approach_position_box = np.asarray(
                approach_position_box,
                dtype=np.float64,
            ).reshape(-1)
            if self.approach_position_box.shape != (4,):
                raise ValueError(
                    "approach_position_box must contain xmin xmax ymin ymax"
                )
            if not np.isfinite(self.approach_position_box).all():
                raise ValueError("approach_position_box must be finite")
        if (
            self.approach_position_box[0]
            > self.approach_position_box[1]
            or self.approach_position_box[2]
            > self.approach_position_box[3]
        ):
            raise ValueError(
                "approach_position_box bounds must be ordered"
            )

        # The reward configuration above always defines the final task:
        # reach within 5 cm while matching the requested direction within the
        # configured final angle.  Training may temporarily use looser active
        # thresholds, but feasibility checking and strict reporting keep using
        # these final values.
        self.curriculum_stage = 6
        self.active_dist_success_thresh = float(
            self.reward_cfg["dist_success_thresh"]
        )
        self.active_orient_success_thresh = float(
            self.reward_cfg["orient_success_thresh"]
        )
        self.active_orient_success_thresh_deg = float(
            np.degrees(self.active_orient_success_thresh)
        )

    def set_approach_curriculum(
        self,
        *,
        stage: int,
        tilt_max_deg: float,
        success_distance_thresh: float,
        success_angle_deg: float,
        position_box: list | tuple | np.ndarray,
    ):
        """Set the active training distribution for the approach task.

        This intentionally does not modify ``reward_cfg``.  The latter is the
        configured final distance/direction specification used by pose-IK
        acceptance and
        strict metrics and stationary pose shaping. These active limits only
        define curriculum milestones, promotion, and rollback statistics.
        """
        if not self.approach_task:
            return
        if int(stage) not in {1, 2, 3, 4, 5, 6}:
            raise ValueError("curriculum stage must be within [1, 6]")
        if not (0.0 <= float(tilt_max_deg) <= 90.0):
            raise ValueError("tilt_max_deg must be within [0, 90]")
        if float(success_distance_thresh) <= 0.0:
            raise ValueError("success_distance_thresh must be positive")
        if not (0.0 < float(success_angle_deg) <= 180.0):
            raise ValueError("success_angle_deg must be within (0, 180]")

        box = np.asarray(position_box, dtype=np.float64).reshape(-1)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError(
                "position_box must contain four finite xmin xmax ymin ymax values"
            )
        if box[0] > box[1] or box[2] > box[3]:
            raise ValueError("position_box bounds must be ordered")

        self.curriculum_stage = int(stage)
        self.target_tilt_max_deg = float(tilt_max_deg)
        self.active_dist_success_thresh = float(success_distance_thresh)
        self.active_orient_success_thresh = float(
            np.deg2rad(success_angle_deg)
        )
        self.active_orient_success_thresh_deg = float(success_angle_deg)
        self.approach_position_box = box.copy()

    # ---------- MJCF setup ----------
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
        if self.approach_task:
            world.add(
                "site",
                name="goal_direction_site",
                type="cylinder",
                size=[0.004, self.goal_direction_length / 2.0],
                rgba=[1.0, 0.8, 0.1, 0.0],
                pos=[0.0, 0.0, 1.0],
                quat=_quat_from_z_axis(self.target_direction).tolist(),
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
        self.goal_direction_sid = self.site_name2id.get(
            "goal_direction_site"
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

    def _update_goal_visuals(self):
        """Move the point and optional approach-direction marker."""
        if self.goal_pos is not None:
            self.model.site_pos[self.goal_sid] = self.goal_pos
        if self.goal_direction_sid is None:
            return
        if self.goal_pos is None:
            self.model.site_rgba[self.goal_direction_sid, 3] = 0.0
            return

        self.model.site_pos[self.goal_direction_sid] = (
            self.goal_pos
            - 0.5 * self.goal_direction_length * self.target_direction
        )
        self.model.site_quat[self.goal_direction_sid] = (
            _quat_from_z_axis(self.target_direction)
        )
        self.model.site_rgba[self.goal_direction_sid] = np.array(
            [1.0, 0.8, 0.1, 0.9],
            dtype=np.float64,
        )

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
                if self.goal_pos is not None and hasattr(self, 'goal_sid'):
                    self.model.site_pos[self.goal_sid] = self.goal_pos
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

    # ---------- Interactive viewer (blocking) ----------
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
                    if self.goal_pos is not None and hasattr(self, 'goal_sid'):
                        self.model.site_pos[self.goal_sid] = self.goal_pos
                    self._pd_to_ctrl(q_hold)
                    mj.mj_step(self.model, self.data)
                    viewer.sync()
        except RuntimeError as e:
            print("Built-in viewer couldn't start:", e)
            return self._fallback_interactive_viewer()


    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_i = 0
        self.pose_success_streak = 0
        self.previous_action = np.zeros(self.nu, dtype=np.float64)
        self.prev_angle_error = None
        self._milestone_20cm_claimed = False
        self._milestone_10cm_claimed = False
        self._active_success_claimed = False
        self.ever_active_success = False

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
        if self.start_pos is not None:
            self.model.site_pos[self.start_sid] = self.start_pos
        self._update_goal_visuals()

        # Goal/direction sampling depending on target_mode. Approach-task
        # feasibility is checked on a scratch MjData, so the live reset state
        # and compiled model are preserved.
        mode = self.target_mode
        if self.approach_task and self.require_pose_feasible:
            self._sample_feasible_approach_target(
                mode,
                self.data.qpos[:self.nq].copy(),
            )
        elif mode == 'fixed':
            if self.target_pos_cli is not None:
                self.goal_pos = np.array(self.target_pos_cli, dtype=np.float64)
            else:
                self.goal_pos = np.array([-0.4, 0.0, 0.1405+0.15], dtype=np.float64)
            self.target_region = "fixed"
            if self.approach_task:
                self._sample_target_direction()
        elif mode == 'random':
            if self.approach_task:
                self.goal_pos, self.target_region = (
                    self._sample_random_goal(self.approach_position_box)
                )
                self._sample_target_direction()
            else:
                self.goal_pos, self.target_region = self._sample_random_goal()
        elif mode == 'none':
            self.goal_pos = None
            self.target_region = "none"
            if self.approach_task:
                self._sample_target_direction()

        if self.start_pos is not None:
            self.model.site_pos[self.start_sid] = self.start_pos
        self._update_goal_visuals()

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

        # Initialize distance used for progress calculation (0 if no goal)

        ee_pos = self._ee_pos()
        if self.goal_pos is not None:
            self.prev_dist = np.linalg.norm(ee_pos - self.goal_pos)
        else:
            self.prev_dist = 0.0
        if self.approach_task:
            self.prev_angle_error = self._direction_angle_error()
            self.min_distance = float(
                self.prev_dist if self.goal_pos is not None else np.inf
            )
            self.min_angle_error = float(self.prev_angle_error)
            if self.pose_task:
                self.min_pose_error = self._keypoint_error(
                    ee_pos,
                    self._ee_direction(),
                )
            else:
                self.min_pose_error = self._normalized_pose_error(
                    self.prev_dist if self.goal_pos is not None else None,
                    self.prev_angle_error,
                )
            self.ever_position_success = bool(
                self.goal_pos is not None
                and self.prev_dist < self.reward_cfg["dist_success_thresh"]
            )
            self.ever_direction_success = bool(
                self.prev_angle_error
                < self.reward_cfg["orient_success_thresh"]
            )
            self.ever_active_success = bool(
                self._is_success(self.prev_dist, self.prev_angle_error)
            )

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = {
            "q_start": self.data.qpos[:self.nq].copy(),
            "active_table": getattr(self, "active_table", None),
            "target_region": self.target_region,
        }
        if self.approach_task:
            info.update({
                "target_direction": self.target_direction.copy(),
                "angle_error": float(self.prev_angle_error),
                "angle_error_deg": float(
                    np.degrees(self.prev_angle_error)
                ),
                "target_tilt_deg": self.target_tilt_deg,
                "target_azimuth_deg": self.target_azimuth_deg,
            })
        return obs, info

    def _sample_target_direction(self):
        """Select the fixed direction or sample one inside a downward cone."""
        if self.target_direction_mode == "fixed":
            self.target_direction = self.fixed_target_direction.copy()
            self.target_tilt_deg = float(
                np.degrees(
                    np.arccos(
                        np.clip(-self.target_direction[2], -1.0, 1.0)
                    )
                )
            )
            self.target_azimuth_deg = float(
                np.degrees(
                    np.arctan2(
                        self.target_direction[1],
                        self.target_direction[0],
                    )
                )
            )
            return self.target_direction.copy()

        self.target_tilt_deg = float(
            self.rng.uniform(
                self.target_tilt_min_deg,
                self.target_tilt_max_deg,
            )
        )
        self.target_azimuth_deg = float(
            self.rng.uniform(
                self.target_azimuth_min_deg,
                self.target_azimuth_max_deg,
            )
        )
        tilt = np.deg2rad(self.target_tilt_deg)
        azimuth = np.deg2rad(self.target_azimuth_deg)
        self.target_direction = _normalize_direction(np.array([
            np.sin(tilt) * np.cos(azimuth),
            np.sin(tilt) * np.sin(azimuth),
            -np.cos(tilt),
        ], dtype=np.float64))
        return self.target_direction.copy()

    def _sample_random_goal(self, box=None):
        """Sample a full-box goal or a diagnosed hard-region mixture."""
        sample_box = self.target_random_box if box is None else box
        xmin, xmax, ymin, ymax = np.asarray(
            sample_box,
            dtype=np.float64,
        ).tolist()
        x_low, x_high = min(xmin, xmax), max(xmin, xmax)
        y_low, y_high = min(ymin, ymax), max(ymin, ymax)

        if self.target_sampling == "uniform":
            region = "uniform"
            x = self.rng.uniform(x_low, x_high)
            y = self.rng.uniform(y_low, y_high)
        else:
            draw = self.rng.random()
            if draw < 0.50:
                region = "uniform"
                x = self.rng.uniform(x_low, x_high)
                y = self.rng.uniform(y_low, y_high)
            elif draw < 0.75:
                region = "hard_negative_y"
                x = self.rng.uniform(x_low, x_high)
                y = self.rng.uniform(y_low, -0.15)
            else:
                region = "hard_far_x"
                x = self.rng.uniform(x_low, -0.45)
                y = self.rng.uniform(y_low, y_high)

        goal = np.array([x, y, 0.2905], dtype=np.float64)
        return goal, region

    @staticmethod
    def _rotation_error(current_direction, target_direction):
        """Axis-angle vector rotating one unit direction onto another."""
        current = _normalize_direction(current_direction)
        target = _normalize_direction(target_direction)
        cross = np.cross(current, target)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
        angle = math.acos(dot)
        if cross_norm > 1e-10:
            return cross / cross_norm * angle
        if dot > 0.0:
            return np.zeros(3, dtype=np.float64)
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(current[0]) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize_direction(np.cross(current, basis))
        return axis * np.pi

    def _solve_pose_ik(
        self,
        target_pos,
        target_direction,
        initial_guess,
    ):
        """Find a collision-free pose satisfying position and tool-axis limits."""
        scratch = mj.MjData(self.model)
        q = np.asarray(initial_guess, dtype=np.float64).copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_direction = _normalize_direction(target_direction)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        joint_range = self.model.jnt_range[:self.nq]
        orientation_scale = 0.25
        damping = 2e-3

        for _ in range(350):
            scratch.qpos[:self.nq] = q
            scratch.qvel[:] = 0.0
            mj.mj_forward(self.model, scratch)
            ee_pos = scratch.site_xpos[self.ee_sid].copy()
            rotation = scratch.site_xmat[self.ee_sid].reshape(3, 3)
            current_direction = rotation[:, 2]
            position_error = target_pos - ee_pos
            angle_error = float(np.arccos(np.clip(
                np.dot(current_direction, target_direction),
                -1.0,
                1.0,
            )))
            if (
                np.linalg.norm(position_error)
                < self.reward_cfg["dist_success_thresh"]
                and angle_error
                < self.reward_cfg["orient_success_thresh"]
                and not self._data_has_collision(scratch)
            ):
                return q

            rotation_error = self._rotation_error(
                current_direction,
                target_direction,
            )
            mj.mj_jacSite(
                self.model,
                scratch,
                jacp,
                jacr,
                self.ee_sid,
            )
            jacobian = np.vstack((
                jacp[:, :self.nq],
                orientation_scale * jacr[:, :self.nq],
            ))
            error = np.concatenate((
                position_error,
                orientation_scale * rotation_error,
            ))
            system = jacobian @ jacobian.T
            dq = jacobian.T @ np.linalg.solve(
                system + (damping ** 2) * np.eye(6),
                error,
            )
            q += 0.35 * np.clip(dq, -0.25, 0.25)
            q = np.clip(q, joint_range[:, 0], joint_range[:, 1])
        return None

    def _sample_feasible_approach_target(self, mode, initial_guess):
        """Sample position/direction pairs until pose IK verifies feasibility."""
        if mode == "none":
            self.goal_pos = None
            self.target_region = "none"
            self._sample_target_direction()
            return

        for _ in range(self.pose_feasibility_attempts):
            if mode == "fixed":
                goal = np.array(
                    self.target_pos_cli
                    if self.target_pos_cli is not None
                    else [-0.4, 0.0, 0.2905],
                    dtype=np.float64,
                )
                region = "fixed"
            else:
                goal, region = self._sample_random_goal(
                    self.approach_position_box
                )
            direction = self._sample_target_direction()
            if self._solve_pose_ik(
                goal,
                direction,
                initial_guess,
            ) is not None:
                self.goal_pos = goal
                self.target_region = region
                return
        raise RuntimeError(
            "Unable to sample a collision-free RM75 approach pose after "
            f"{self.pose_feasibility_attempts} attempts; reduce the target "
            "box or tilt range."
        )


    def _pd_to_ctrl(self, q_des):
        """Send desired joint angles to the seven position actuators."""
        self.data.ctrl[:self.nu] = q_des[:self.nu]

    def _step_sim(self, n=1, hold_target=True):
        for _ in range(n):
            if hold_target and self.q_target is not None:
                self._pd_to_ctrl(self.q_target)
            mj.mj_step(self.model, self.data)

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

        # Update goal/start sites for visualization
        self._update_goal_visuals()
        if self.start_pos is not None:
            self.model.site_pos[self.start_sid] = self.start_pos
        self._step_sim(self.substeps)

        if self.render_mode == "human":
            self.render()

        # Collision check
        collided = self._check_collision()

        # Compute goal distance and success flag
        ee_pos = self._ee_pos()
        cfg = self.reward_cfg
        if self.goal_pos is not None:
            dist = np.linalg.norm(ee_pos - self.goal_pos)
        else:
            dist = None
        current_direction = (
            self._ee_direction()
            if self.approach_task
            else None
        )
        angle_error = (
            self._direction_angle_error(current_direction)
            if self.approach_task
            else None
        )
        # Active success is a curriculum milestone. Only strict final-pose
        # success ends an approach episode, allowing the policy to learn from
        # the active boundary all the way to the final target.
        active_success = self._is_success(dist, angle_error)
        within_strict_tolerance = self._is_strict_success(
            dist,
            angle_error,
        )
        if self.pose_task:
            terminal_success = self._update_pose_success_streak(
                within_strict_tolerance and not collided
            )
            strict_success = terminal_success
            active_success = terminal_success
        else:
            strict_success = within_strict_tolerance
            terminal_success = bool(
                strict_success if self.approach_task else active_success
            )
        position_success = bool(
            dist is not None and dist < cfg["dist_success_thresh"]
        )
        direction_success = bool(
            not self.approach_task
            or (
                angle_error is not None
                and angle_error < cfg["orient_success_thresh"]
            )
        )

        if self.approach_task:
            self.ever_active_success = bool(
                self.ever_active_success or active_success
            )
        active_success_bonus = (
            0.0
            if self.pose_task
            else self._claim_active_success_bonus(active_success)
        )

        truncated = self.step_i >= self.horizon
        timed_out = bool(
            truncated and not terminal_success and not collided
        )
        milestone_reward = (
            0.0
            if self.pose_task
            else self._compute_milestone_reward(dist, angle_error)
        )

        if self.pose_task:
            keypoint_error = self._keypoint_error(
                ee_pos,
                current_direction,
            )
            reward, reward_components = self._compute_pose_reward(
                dist=dist,
                prev_dist=self.prev_dist,
                angle_error=angle_error,
                prev_angle_error=self.prev_angle_error,
                action=normalized_action,
                previous_action=self.previous_action,
                collided=collided,
                success=terminal_success,
            )
            self.previous_action = normalized_action.copy()
        else:
            keypoint_error = None
            reward, reward_components = self._compute_reward(
                dist=dist,
                prev_dist=self.prev_dist,
                collided=collided,
                success=terminal_success,
                timed_out=timed_out,
                milestone_reward=milestone_reward,
                active_success_bonus=active_success_bonus,
                angle_error=angle_error,
                prev_angle_error=self.prev_angle_error,
            )
        self.prev_dist = dist if dist is not None else 0.0
        self.prev_angle_error = angle_error

        if self.approach_task:
            self.min_distance = min(self.min_distance, float(dist))
            self.min_angle_error = min(
                self.min_angle_error,
                float(angle_error),
            )
            current_pose_error = (
                keypoint_error
                if self.pose_task
                else self._normalized_pose_error(dist, angle_error)
            )
            self.min_pose_error = min(
                self.min_pose_error,
                current_pose_error,
            )
            self.ever_position_success = bool(
                self.ever_position_success or position_success
            )
            self.ever_direction_success = bool(
                self.ever_direction_success or direction_success
            )

        terminated = bool(terminal_success or collided)

        obs = self._get_obs()
        info = {
            "success": (
                terminal_success
                if self.pose_task
                else (
                    self.ever_active_success
                    if self.approach_task
                    else terminal_success
                )
            ),
            "collision": collided,
            "timeout": timed_out,
            "dist": dist,
            "target_region": self.target_region,
            "reward_components": reward_components,
            "active_table": getattr(self, "active_table", None),
        }
        if self.approach_task:
            info.update({
                "position_success": position_success,
                "direction_success": direction_success,
                "active_success": active_success,
                "ever_active_success": self.ever_active_success,
                "strict_success": strict_success,
                "curriculum_stage": self.curriculum_stage,
                "active_dist_success_thresh": (
                    self.active_dist_success_thresh
                ),
                "active_orient_success_thresh_deg": (
                    self.active_orient_success_thresh_deg
                ),
                "target_direction": self.target_direction.copy(),
                "angle_error": float(angle_error),
                "angle_error_deg": float(np.degrees(angle_error)),
                "target_tilt_deg": self.target_tilt_deg,
                "target_azimuth_deg": self.target_azimuth_deg,
                "min_distance": self.min_distance,
                "min_angle_error_deg": float(
                    np.degrees(self.min_angle_error)
                ),
                "min_pose_error": self.min_pose_error,
                "ever_position_success": self.ever_position_success,
                "ever_direction_success": self.ever_direction_success,
                "simultaneous_success": active_success,
            })
            if self.pose_task:
                info.update({
                    "within_pose_tolerance": within_strict_tolerance,
                    "pose_success_streak": self.pose_success_streak,
                    "keypoint_error": keypoint_error,
                })
        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        dist,
        prev_dist,
        collided: bool,
        success: bool,
        timed_out: bool = False,
        milestone_reward: float = 0.0,
        active_success_bonus: float = 0.0,
        angle_error: float | None = None,
        prev_angle_error: float | None = None,
    ):
        """
        Centralized reward computation.
        If dist is None, no distance-based terms are applied. Timeout is a
        terminal failure only when the horizon is reached without collision
        or success.
        """
        cfg = self.reward_cfg

        # Point task: distance-only shaping remains unchanged for legacy RM75.
        # Approach task: the Chebyshev-style pose error is governed by the
        # worse normalized constraint, so distance cannot compensate for a
        # badly aligned tool axis (or vice versa).
        progress = 0.0
        pose_progress = 0.0
        pose_penalty = 0.0
        if (
            self.approach_task
            and dist is not None
            and angle_error is not None
            and prev_angle_error is not None
            and prev_dist != 0.0
        ):
            current_pose_error = self._normalized_pose_error(
                dist,
                angle_error,
            )
            previous_pose_error = self._normalized_pose_error(
                prev_dist,
                prev_angle_error,
            )
            pose_progress = cfg["pose_progress_scale"] * (
                previous_pose_error - current_pose_error
            )
            pose_penalty = (
                -cfg["pose_error_weight"] * current_pose_error
            )
        elif dist is not None and prev_dist != 0.0:
            progress = cfg["progress_scale"] * (prev_dist - dist)

        # Distance penalty (absolute distance to goal)
        dist_penalty = 0.0
        if dist is not None and not self.approach_task:
            dist_penalty = -cfg["dist_weight"] * dist

        reward = 0.0
        reward += progress
        reward += pose_progress
        reward += pose_penalty
        reward += dist_penalty
        reward -= cfg["time_penalty"]
        reward += milestone_reward
        reward += active_success_bonus

        # Legacy optional orientation shaping for the point task. The approach
        # task is fully handled by the joint pose potential above.
        orient_reward = 0.0
        if (
            not self.approach_task
            and self.ee_sid is not None
            and cfg["orient_weight"] != 0.0
        ):
            try:
                R = self.data.site_xmat[self.ee_sid].reshape(3, 3)
                axis = (self.orient_axis or "z").lower().strip()
                sign = -1.0 if axis.startswith("-") else 1.0
                axis = axis[1:] if axis.startswith("-") else axis
                idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
                v = R[:, idx] * sign
                align = float(np.dot(v, np.array([0.0, 0.0, -1.0])))
                orient_reward = cfg["orient_weight"] * (align ** cfg["orient_power"])
                reward += orient_reward
            except Exception:
                pass

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
            "milestone": milestone_reward,
            "orient": orient_reward,
            "collision": collision_reward,
            "success": success_reward,
            "timeout": timeout_reward,
        }
        if self.approach_task:
            components.update({
                "pose_progress": pose_progress,
                "pose_penalty": pose_penalty,
                "active_success": active_success_bonus,
            })
        return reward, components

    def _keypoint_error(self, ee_pos=None, current_direction=None):
        """Return one position-and-direction error in metres."""
        if self.goal_pos is None:
            return float("inf")
        position = self._ee_pos() if ee_pos is None else np.asarray(
            ee_pos,
            dtype=np.float64,
        )
        direction = (
            self._ee_direction()
            if current_direction is None
            else _normalize_direction(
                current_direction,
                name="current_direction",
            )
        )
        current_tip = position + self.pose_keypoint_length * direction
        target_tip = (
            self.goal_pos
            + self.pose_keypoint_length * self.target_direction
        )
        return float(0.5 * (
            np.linalg.norm(position - self.goal_pos)
            + np.linalg.norm(current_tip - target_tip)
        ))

    def _compute_pose_reward(
        self,
        *,
        dist,
        prev_dist,
        angle_error,
        prev_angle_error,
        action,
        previous_action,
        collided,
        success,
    ):
        """Reward pose progress without making early termination attractive.

        Position dominates far from the goal.  A smooth distance gate raises
        the orientation weight near the goal, while the difference of one
        combined potential makes closed state loops cancel instead of earning
        reward through oscillation.
        """
        dist = float(dist)
        prev_dist = float(prev_dist)
        angle_error = float(angle_error)
        prev_angle_error = float(prev_angle_error)
        action = np.asarray(action, dtype=np.float64)
        previous_action = np.asarray(previous_action, dtype=np.float64)

        def orientation_gate(distance):
            exponent = np.clip((distance - 0.20) / 0.04, -60.0, 60.0)
            return 0.1 + 0.9 / (1.0 + np.exp(exponent))

        distance_scale = self.reward_cfg["dist_success_thresh"]
        angle_scale = self.reward_cfg["orient_success_thresh"]
        previous_gate = orientation_gate(prev_dist)
        current_gate = orientation_gate(dist)

        position_progress = (prev_dist - dist) / distance_scale
        previous_orientation_cost = (
            0.5 * previous_gate * prev_angle_error / angle_scale
        )
        current_orientation_cost = (
            0.5 * current_gate * angle_error / angle_scale
        )
        orientation_progress = (
            previous_orientation_cost - current_orientation_cost
        )
        pose_progress = position_progress + orientation_progress

        action_term = -0.002 * float(np.sum(action ** 2))
        smoothness_term = -0.003 * float(
            np.sum((action - previous_action) ** 2)
        )
        time_term = -0.001
        collision_term = (
            -self.reward_cfg["collision_penalty"] if collided else 0.0
        )
        success_term = 20.0 if success else 0.0
        reward = (
            pose_progress
            + action_term
            + smoothness_term
            + time_term
            + collision_term
            + success_term
        )
        return reward, {
            "position_progress": position_progress,
            "orientation_progress": orientation_progress,
            "pose_progress": pose_progress,
            "action": action_term,
            "smoothness": smoothness_term,
            "time": time_term,
            "collision": collision_term,
            "success": success_term,
        }

    def _update_pose_success_streak(self, within_tolerance):
        """Require a stable final pose for consecutive control steps."""
        if within_tolerance:
            self.pose_success_streak += 1
        else:
            self.pose_success_streak = 0
        return self.pose_success_streak >= self.pose_success_hold_steps

    def _normalized_pose_error(self, dist, angle_error):
        """Return a smooth, jointly position-and-direction pose error.

        A plain ``max(distance_error, angle_error)`` is unsafe for learning:
        when both errors are similar, improving the non-maximum term produces
        no reward change.  A centered smooth maximum preserves the meaning of
        the worse constraint.  Explicit position and direction terms retain
        useful gradients for both, with extra weight on the observed position
        bottleneck.
        """
        if dist is None or angle_error is None:
            return float("inf")
        distance_error = (
            float(dist) / self.reward_cfg["dist_success_thresh"]
        )
        direction_error = (
            float(angle_error) / self.reward_cfg["orient_success_thresh"]
        )
        beta = 2.0
        smooth_max = (
            np.logaddexp(beta * distance_error, beta * direction_error)
            / beta
            - np.log(2.0) / beta
        )
        # Position remains the primary bottleneck in the combined task.  The
        # asymmetric terms prevent a policy from improving orientation while
        # allowing target distance to drift backwards.
        return float(
            0.5 * smooth_max
            + 0.45 * distance_error
            + 0.05 * direction_error
        )

    def _direction_angle_error(self, current_direction=None):
        """Return the unsigned angle to the desired approach direction."""
        current = (
            self._ee_direction()
            if current_direction is None
            else _normalize_direction(
                current_direction,
                name="current_direction",
            )
        )
        cosine = float(np.clip(
            np.dot(current, self.target_direction),
            -1.0,
            1.0,
        ))
        return float(np.arccos(cosine))

    def _is_success(self, dist, angle_error=None):
        """Apply active point-only or curriculum pose success rules."""
        distance_thresh = (
            self.active_dist_success_thresh
            if self.approach_task
            else self.reward_cfg["dist_success_thresh"]
        )
        if dist is None or dist >= distance_thresh:
            return False
        if not self.approach_task:
            return True
        return bool(
            angle_error is not None
            and angle_error < self.active_orient_success_thresh
        )

    def _is_strict_success(self, dist, angle_error=None):
        """Apply the final task criteria, independent of curriculum stage."""
        if dist is None or dist >= self.reward_cfg["dist_success_thresh"]:
            return False
        if not self.approach_task:
            return True
        return bool(
            angle_error is not None
            and angle_error < self.reward_cfg["orient_success_thresh"]
        )

    def _claim_active_success_bonus(self, active_success):
        """Award one curriculum milestone without ending the episode."""
        if (
            not self.approach_task
            or self.curriculum_stage >= 6
            or not active_success
            or self._active_success_claimed
        ):
            return 0.0
        self._active_success_claimed = True
        return float(self.reward_cfg["active_success_bonus"])

    def _compute_distance_milestone_reward(self, dist):
        """Award each distance milestone at most once in an episode."""
        if dist is None:
            return 0.0

        reward = 0.0
        if dist < 0.20 and not self._milestone_20cm_claimed:
            reward += self.reward_cfg["milestone_20cm_bonus"]
            self._milestone_20cm_claimed = True
        if dist < 0.10 and not self._milestone_10cm_claimed:
            reward += self.reward_cfg["milestone_10cm_bonus"]
            self._milestone_10cm_claimed = True
        return reward

    def _compute_milestone_reward(self, dist, angle_error=None):
        """Award point milestones or joint pose milestones once per episode."""
        if not self.approach_task:
            return self._compute_distance_milestone_reward(dist)
        if dist is None or angle_error is None:
            return 0.0

        reward = 0.0
        if (
            dist < 0.20
            and angle_error < np.deg2rad(45.0)
            and not self._milestone_20cm_claimed
        ):
            reward += self.reward_cfg[
                "approach_milestone_20cm_bonus"
            ]
            self._milestone_20cm_claimed = True
        if (
            dist < 0.075
            and angle_error < np.deg2rad(25.0)
            and not self._milestone_10cm_claimed
        ):
            reward += self.reward_cfg[
                "approach_precision_milestone_bonus"
            ]
            self._milestone_10cm_claimed = True
        return reward

    def _ee_pos(self):
        if self.ee_sid is None:
            return np.zeros(3)
        mj.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.ee_sid].copy()

    def _ee_direction(self):
        """Return the end-effector site's local +Z axis in world coordinates."""
        if self.ee_sid is None:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        mj.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.ee_sid].reshape(3, 3)
        return _normalize_direction(
            rotation[:, 2],
            name="end-effector direction",
        )

    def _get_obs(self):
        if self.pose_task:
            return self._get_pose_observation()

        q = self.data.qpos[:self.nq].copy()
        qd = self.data.qvel[:self.nq].copy()
        if self.q_target is None or self.q_target.shape != (self.nq,):
            tracking_error = np.zeros(self.nq, dtype=np.float64)
        else:
            tracking_error = self.q_target - q
        ee = self._ee_pos()
        goal = self.goal_pos if self.goal_pos is not None else np.zeros(3)
        coll = float(self._check_collision())
        parts = [
            q,
            qd,
            tracking_error,
            ee.astype(np.float64),
            goal.astype(np.float64),
        ]
        if self.approach_task:
            current_direction = self._ee_direction()
            parts.extend([
                current_direction,
                self.target_direction,
            ])
        parts.append(np.array([coll], dtype=np.float64))
        obs = np.concatenate(parts).astype(np.float32)
        return obs

    def _get_pose_observation(self):
        """Return the compact, physically normalized pose observation."""
        q = self.data.qpos[:self.nq].copy()
        qd = self.data.qvel[:self.nq].copy()
        joint_range = self.model.jnt_range[:self.nq]
        joint_center = 0.5 * (joint_range[:, 0] + joint_range[:, 1])
        joint_half_range = 0.5 * (
            joint_range[:, 1] - joint_range[:, 0]
        )
        safe_half_range = np.where(
            joint_half_range > 1e-9,
            joint_half_range,
            1.0,
        )
        normalized_q = np.clip(
            (q - joint_center) / safe_half_range,
            -1.0,
            1.0,
        )
        normalized_qd = np.clip(
            qd / self.pose_velocity_scale,
            -1.0,
            1.0,
        )
        if self.goal_pos is None:
            position_error = np.zeros(3, dtype=np.float64)
            rotation_error = np.zeros(3, dtype=np.float64)
        else:
            position_error = np.clip(
                (self.goal_pos - self._ee_pos())
                / self.pose_position_error_scale,
                -2.0,
                2.0,
            )
            rotation_error = np.clip(
                self._rotation_error(
                    self._ee_direction(),
                    self.target_direction,
                ) / np.pi,
                -1.0,
                1.0,
            )
        return np.concatenate((
            normalized_q,
            normalized_qd,
            position_error,
            rotation_error,
        )).astype(np.float32)

    def _solve_ik(self, target_pos: np.ndarray, initial_guess: np.ndarray) -> np.ndarray | None:
        """Simple IK solver using damped least squares.
        Returns joint configuration or None if failed.
        """
        q = initial_guess.copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        damping = 1e-3
        lamI3 = (damping ** 2) * np.eye(3)

        jrange = self.model.jnt_range[:self.nq]

        for it in range(500):  # max iterations
            self.data.qpos[:self.nq] = q
            self.data.qvel[:] = 0.0
            mj.mj_forward(self.model, self.data)

            ee_pos = self.data.site_xpos[self.ee_sid].copy()
            e_pos = target_pos - ee_pos

            if np.linalg.norm(e_pos) < self.ik_tolerance:
                return q

            mj.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_sid)
            Jp = jacp[:, :self.nq]

            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + lamI3, e_pos)
            dq = np.clip(dq, -0.2, 0.2)

            q = q + 0.2 * dq
            # Clip to joint limits
            for j in range(self.nq):
                lo, hi = jrange[j]
                if np.isfinite(lo) and np.isfinite(hi):
                    q[j] = np.clip(q[j], lo, hi)

        # Failed to converge
        return None

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

    def _check_collision(self):
        mj.mj_forward(self.model, self.data)
        return self._data_has_collision(self.data)

    def _data_has_collision(self, data):
        """Classify contacts from live or scratch MuJoCo data."""
        for i in range(data.ncon):
            c = data.contact[i]
            if self._contact_pair_is_collision(c.geom1, c.geom2):
                return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            # Off-screen rendering for video recording
            if not hasattr(self, "_offscreen_renderer"):
                from mujoco import Renderer
                # HD resolution: 1280x720 (instead of 640x480)
                self._offscreen_renderer = Renderer(self.model, height=480, width=1280)

            # Update visualization sites
            if self.start_pos is not None:
                self.model.site_pos[self.start_sid] = self.start_pos
            self._update_goal_visuals()

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
            del self._offscreen_renderer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Open a blocking viewer; close the window to exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", type=str, default="random",
                        help="How to set goal position: fixed|random|none (random uses built-in region)")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None)
    args = parser.parse_args()

    env = RM75P2PEnv(
        render_mode=None,
        seed=args.seed,
        target_mode=args.target_mode,
        target_pos=args.target_pos,
    )

    if args.interactive:
        env.reset()
        env.interactive_viewer()
        env.close()
    else:
        obs, info = env.reset()
        done = False
        total = 0.0
        while not done:
            a = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        print("episode done. total reward:", total)
