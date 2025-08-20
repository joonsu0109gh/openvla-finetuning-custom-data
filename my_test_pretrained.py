import os
import time
import logging
import traceback
import numpy as np
import cv2
import torch

import pyrealsense2 as rs

from pathlib import Path
from typing import Any, Dict, Union
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.transform import Rotation as R

from franky import Robot, JointMotion, ReferenceType, Affine, CartesianMotion, Gripper
import threading
import json
import argparse

# ------------------------------ Constants ------------------------------
CONTROL_LOOP_RATE_HZ =5

GRIPPER_OPEN_WIDTH = 0.08
GRIPPER_CLOSED_WIDTH = 0.02

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


# ------------------------------ Utilities ------------------------------
def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in str(openvla_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# ------------------------------ Vision-Language Controller ------------------------------
class OpenVLAController:
    def __init__(self, openvla_path: Union[str, Path], attn_impl: str = "flash_attention_2"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)

        self.model = AutoModelForVision2Seq.from_pretrained(
            openvla_path,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        self.openvla_path = openvla_path


        
        dataset_statistics_path = os.path.join(openvla_path, "dataset_statistics.json")

        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                norm_stats = json.load(f)
            self.model.norm_stats = norm_stats
        else:
            raise FileNotFoundError(
                f"Dataset statistics file not found at {dataset_statistics_path}. "
                "Make sure to run the training script with --save_dataset_statistics."
            )
        
        self.unnorm_key = "my_dataset"  

        print(f"✅ OpenVLA model loaded from {openvla_path}")

    def predict_action(self, image: np.ndarray, instruction: str) -> Dict[str, Any]:
        try:
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                self.device, dtype=torch.bfloat16
            )

            output = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)

            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()

            if isinstance(output, np.ndarray) and output.size == 7:
                return {
                    "dpos_x": output[0],
                    "dpos_y": output[1],
                    "dpos_z": output[2],
                    "drot_x": output[3],
                    "drot_y": output[4],
                    "drot_z": output[5],
                    "grip_command": "open" if output[6] > 0.5 else "close",
                }

            return output
        except Exception:
            logging.error(traceback.format_exc())
            return {"error": "VLA prediction failed"}


# ------------------------------ Camera Manager ------------------------------
class CameraManager:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.started = False
        self.width = width
        self.height = height

    def start(self, warmup_frames: int = 10):

        self.pipeline.start(self.config)
        self.started = True
        self.pipeline.wait_for_frames(1000)

    def get_frame(self) -> Union[np.ndarray, None]:
        if not self.started:
            self.start()
        try:
            frames = self.pipeline.wait_for_frames()  
            color_frame = frames.get_color_frame()

            image = np.asanyarray(color_frame.get_data())

            return image

        except Exception as e:
            logging.error(f"CameraManager.get_frame error: {e}")
            return None

    def restart(self):
        self.stop()
        time.sleep(0.2)
        try:
            self.start(warmup_frames=5)
        except Exception as e:
            logging.error(f"CameraManager.restart failed: {e}")

    def stop(self):
        if not self.started:
            return
        try:
            self.pipeline.stop()
        except Exception:
            pass
        finally:
            self.started = False


# ------------------------------ Robot Controller ------------------------------
class RobotController:
    def __init__(self, hostname: str):
        self.robot = Robot(hostname)
        self.gripper = Gripper(hostname)
        self.robot.relative_dynamics_factor = 0.075
        self.robot.recover_from_errors()
        self.previous_gripper_state = None

    def move_home(self):
        home_pose = [-0.0026, -0.7855, 0.0011, -2.3576, 0.0038, 1.5738, 0.7780]

        self.robot.move(JointMotion(home_pose, ReferenceType.Absolute), asynchronous=True)
        self.robot.join_motion(timeout=10.0)
        self.gripper.move_async(width=GRIPPER_OPEN_WIDTH, speed=0.1)

    def apply_action(self, action: Dict[str, Any]):
        dpos = np.array([action['dpos_x'], action['dpos_y'], action['dpos_z']])
        drot = R.from_euler("xyz", [action['drot_x'], action['drot_y'], action['drot_z']]).as_quat()

        print(f"Applying action: dpos={dpos}, drot={drot}, grip_command={action.get('grip_command')}")
        
        current_pose = self.robot.state.O_T_EE
        ee_pos = current_pose.translation
        ee_ori = current_pose.quaternion

        target_pos = ee_pos + dpos
        target_ori = R.from_quat(ee_ori) * R.from_quat(drot)
        target_ori = target_ori.as_quat()

        motion = CartesianMotion(Affine(target_pos, target_ori), ReferenceType.Absolute)
        self.robot.move(motion, asynchronous=True)

        # self.robot.join_motion(timeout=10.0)

        if action.get("grip_command") != self.previous_gripper_state:
            if action.get("grip_command") == "close":
                if self.gripper.width > (GRIPPER_OPEN_WIDTH / 2):
                    threading.Thread(
                        target=self.gripper_close, daemon=True
                    ).start()

            elif action.get("grip_command") == "open":
                threading.Thread(
                        target=self.gripper_open, daemon=True
                    ).start()
        else:
            print("No gripper action needed, skipping.")

        self.previous_gripper_state = action.get("grip_command")

    def gripper_open(self):
        self.gripper.move_async(width=GRIPPER_OPEN_WIDTH, speed=0.05)
    def gripper_close(self):
        self.gripper.grasp_async(
                    width=GRIPPER_CLOSED_WIDTH, speed=0.05, force=20,
                    epsilon_inner=0.1, epsilon_outer=0.1
                )

# ------------------------------ Main Loop ------------------------------
def main():

    parser = argparse.ArgumentParser(description="Openvla franka deployment.")
    parser.add_argument('--checkpoint-path', type=str, default="", help="path to the OpenVLA checkpoint")
    parser.add_argument('--instruction', type=str, default="", help="Instruction for the robot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    hostname = "172.16.0.2"
    instruction = args.instruction

    camera = None
    robot = None

    try:
        robot = RobotController(hostname)
        robot.move_home()
        print("✅ Robot and gripper initialized.")

        camera = CameraManager()
        camera.start(warmup_frames=10)
        print("✅ Camera started. Waiting for frames...")
        time.sleep(1)

        model_path = args.checkpoint_path

        controller = OpenVLAController(model_path)
        print(f"📌 Instruction: {instruction}")

        control_period = 1.0 / CONTROL_LOOP_RATE_HZ
        next_iteration_time = time.time()


        while True:
            image = camera.get_frame()

            action = controller.predict_action(image, instruction)

            robot.apply_action(action)

            next_iteration_time += control_period
            sleep_time = next_iteration_time - time.time()
            if sleep_time > 0:
                if sleep_time < 0.0005:
                    while time.time() < next_iteration_time:
                        pass
                else:
                    time.sleep(sleep_time)
            else:
                next_iteration_time = time.time()

    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
        logging.error(traceback.format_exc())
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if camera is not None:
                camera.stop()   
        except Exception:
            pass
        try:
            if robot is not None:
                robot.gripper.move_async(width=GRIPPER_OPEN_WIDTH, speed=0.05)
        except Exception:
            pass
        print("✅ Cleanup completed. Exiting.")


if __name__ == "__main__":
    main()

# sudo -E env LD_LIBRARY_PATH=$LD_LIBRARY_PATH $(which python) my_test_pretrained.py
