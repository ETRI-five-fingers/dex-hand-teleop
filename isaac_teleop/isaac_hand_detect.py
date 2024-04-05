import cv2
import numpy as np
import rclpy

from hand_detector.hand_monitor import Record3DSingleHandMotionControl
from isaac_allegro import IsaacAllegro


def main():
    # IsaacSim ROS2 node
    rclpy.init(args=None)
    isaac_allegro = IsaacAllegro()

    # Perception
    motion_control = Record3DSingleHandMotionControl(hand_mode="right_hand", show_hand=True, device="cuda:1")
    rgb, depth = motion_control.camera.fetch_rgb_and_depth()

    while isaac_allegro.is_connected:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("rgb", bgr)
        if cv2.waitKey(1) == 27:
            break

        if not motion_control.initialized:
            success, motion_data = motion_control.step()
            rgb = motion_data["rgb"]
            if not success:
                continue

            # vertices: (778, 3), faces: (1538, 3)
            print("motion_data ", motion_data["vertices"].shape, motion_data["faces"].shape)

        else:
            success, motion_data = motion_control.step()
            rgb = motion_data["rgb"]

            if not success:
                continue

            root_joint_qpos = motion_control.compute_operator_space_root_qpos(motion_data)
            root_joint_qpos *= 1.0
            mat = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            root_joint_qpos[:3] = mat @ root_joint_qpos[:3]


if __name__ == "__main__":
    print("isaac hand detection module")
    main()
