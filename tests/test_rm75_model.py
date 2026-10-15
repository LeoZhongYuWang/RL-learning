import unittest
from pathlib import Path

import mujoco as mj
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets" / "rm75"
LOCAL_URDF_PATH = ASSET_DIR / "RM75.urdf"
IMPORTED_MODEL_PATH = ASSET_DIR / "rm75_imported.xml"
MODEL_PATH = ASSET_DIR / "rm75.xml"
SCENE_PATH = ASSET_DIR / "scene.xml"

JOINT_NAMES = [f"joint_{index}" for index in range(1, 8)]
ACTUATOR_NAMES = [f"joint_{index}_pos" for index in range(1, 8)]
JOINT_LIMITS = np.array(
    [
        [-3.106, 3.106],
        [-2.2689, 2.2689],
        [-3.106, 3.106],
        [-2.356, 2.356],
        [-3.106, 3.106],
        [-2.234, 2.234],
        [-6.28, 6.28],
    ],
    dtype=np.float64,
)
FORCE_LIMITS = np.array(
    [60.0, 60.0, 30.0, 30.0, 10.0, 10.0, 10.0],
    dtype=np.float64,
)
HOME_QPOS = np.array(
    [
        -0.003093,
        0.447318,
        0.003770,
        1.458466,
        -0.000539,
        0.552761,
        0.0,
    ],
    dtype=np.float64,
)


def object_names(model: mj.MjModel, object_type, count: int) -> list[str]:
    return [
        mj.mj_id2name(model, object_type, index)
        for index in range(count)
    ]


class RM75ModelTests(unittest.TestCase):
    def test_local_urdf_has_relative_mesh_paths(self):
        urdf_text = LOCAL_URDF_PATH.read_text(encoding="utf-8")
        self.assertNotIn("package://", urdf_text)
        for mesh_name in ["base_link", *[f"link_{i}" for i in range(1, 8)]]:
            self.assertTrue(
                (ASSET_DIR / "meshes" / f"{mesh_name}.STL").is_file()
            )

    def test_converted_rm75_model_has_seven_joints(self):
        model = mj.MjModel.from_xml_path(str(IMPORTED_MODEL_PATH))
        self.assertEqual((model.nq, model.nv, model.njnt), (7, 7, 7))
        self.assertEqual(
            object_names(model, mj.mjtObj.mjOBJ_JOINT, model.njnt),
            JOINT_NAMES,
        )
        self.assertEqual(model.nu, 0)

    def test_rm75_control_tcp_and_home_contract(self):
        model = mj.MjModel.from_xml_path(str(MODEL_PATH))
        self.assertEqual((model.nq, model.nv, model.njnt), (7, 7, 7))
        self.assertEqual(model.nu, 7)
        self.assertEqual(
            object_names(model, mj.mjtObj.mjOBJ_ACTUATOR, model.nu),
            ACTUATOR_NAMES,
        )

        np.testing.assert_allclose(model.jnt_range, JOINT_LIMITS)
        np.testing.assert_allclose(model.actuator_ctrlrange, JOINT_LIMITS)
        np.testing.assert_allclose(
            model.actuator_forcerange[:, 0],
            -FORCE_LIMITS,
        )
        np.testing.assert_allclose(
            model.actuator_forcerange[:, 1],
            FORCE_LIMITS,
        )

        ee_site_id = mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_SITE,
            "ee_site",
        )
        self.assertGreaterEqual(ee_site_id, 0)
        ee_body_id = int(model.site_bodyid[ee_site_id])
        self.assertEqual(
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, ee_body_id),
            "link_7",
        )

        home_key_id = mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_KEY,
            "home",
        )
        self.assertGreaterEqual(home_key_id, 0)
        np.testing.assert_allclose(model.key_qpos[home_key_id], HOME_QPOS)
        np.testing.assert_allclose(model.key_ctrl[home_key_id], HOME_QPOS)

    def test_rm75_scene_loads_with_floor(self):
        model = mj.MjModel.from_xml_path(str(SCENE_PATH))
        self.assertEqual(model.nq, 7)
        self.assertEqual(model.nu, 7)
        floor_id = mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_GEOM,
            "floor",
        )
        self.assertGreaterEqual(floor_id, 0)
        self.assertGreaterEqual(model.nlight, 1)


if __name__ == "__main__":
    unittest.main()
