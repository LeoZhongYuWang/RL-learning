#!/usr/bin/env python3
"""Copy the RM75 URDF assets locally and export MuJoCo's imported MJCF.

The ROS workspace is treated as read-only. This script only writes under
``assets/rm75`` in this project. Re-running it refreshes ``RM75.urdf``, the
eight STL files, and ``rm75_imported.xml``; it never overwrites the manually
reviewed ``rm75.xml`` or ``scene.xml`` files.
"""

from pathlib import Path
import shutil

import mujoco as mj


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(
    "/home/zhongyu/Realman_Control/Rl_ws/src/"
    "harvest_arm_rm/urdf/urdf_rm75"
)
DEST_DIR = PROJECT_ROOT / "assets" / "rm75"
PACKAGE_PREFIX = "package://harvest_arm_rm/urdf/urdf_rm75/"
MESH_FILENAMES = [
    "base_link.STL",
    *[f"link_{index}.STL" for index in range(1, 8)],
]
EXPECTED_JOINT_NAMES = [
    f"joint_{index}"
    for index in range(1, 8)
]


def copy_source_assets() -> Path:
    """Create the local URDF copy and copy the eight referenced meshes."""
    source_urdf = SOURCE_DIR / "RM75.urdf"
    source_mesh_dir = SOURCE_DIR / "meshes"
    if not source_urdf.is_file():
        raise FileNotFoundError(source_urdf)
    if not source_mesh_dir.is_dir():
        raise FileNotFoundError(source_mesh_dir)

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    destination_mesh_dir = DEST_DIR / "meshes"
    destination_mesh_dir.mkdir(parents=True, exist_ok=True)

    for mesh_filename in MESH_FILENAMES:
        source_mesh = source_mesh_dir / mesh_filename
        if not source_mesh.is_file():
            raise FileNotFoundError(source_mesh)
        shutil.copy2(source_mesh, destination_mesh_dir / mesh_filename)

    urdf_text = source_urdf.read_text(encoding="utf-8")
    if PACKAGE_PREFIX not in urdf_text:
        raise RuntimeError(
            f"Expected ROS package prefix is absent: {PACKAGE_PREFIX}"
        )
    local_urdf_text = urdf_text.replace(PACKAGE_PREFIX, "")
    if "package://" in local_urdf_text:
        raise RuntimeError(
            "The copied RM75 URDF still contains unresolved package:// paths."
        )

    local_urdf = DEST_DIR / "RM75.urdf"
    local_urdf.write_text(local_urdf_text, encoding="utf-8")
    return local_urdf


def validate_imported_model(model: mj.MjModel) -> None:
    """Fail early if MuJoCo did not import the expected seven-axis robot."""
    dimensions = (model.nq, model.nv, model.njnt, model.nu)
    if dimensions != (7, 7, 7, 0):
        raise RuntimeError(
            "Unexpected RM75 imported model dimensions: "
            f"nq={model.nq}, nv={model.nv}, "
            f"njnt={model.njnt}, nu={model.nu}"
        )

    joint_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]
    if joint_names != EXPECTED_JOINT_NAMES:
        raise RuntimeError(
            "Unexpected RM75 joint order: "
            f"{joint_names}"
        )


def main() -> None:
    local_urdf = copy_source_assets()
    model = mj.MjModel.from_xml_path(str(local_urdf))
    validate_imported_model(model)

    imported_model_path = DEST_DIR / "rm75_imported.xml"
    mj.mj_saveLastXML(str(imported_model_path), model)
    print(f"Local URDF: {local_urdf}")
    print(f"Imported MJCF: {imported_model_path}")
    print("RM75 import verified: nq=7, nv=7, njnt=7, nu=0")


if __name__ == "__main__":
    main()
