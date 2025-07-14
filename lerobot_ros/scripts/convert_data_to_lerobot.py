import uuid
from pathlib import Path
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import List
from PIL import Image as PILImage
from datasets import Dataset, Features, Image, Sequence, Value
import imitator.utils.file_utils as FileUtils

from build_data_from_rosbag import RosbagEpisode
from pathlib import Path
import pickle

import os
import argparse
import datetime

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import imageio
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig

def convert_to_lerobot_v21(
        episode_list: List[RosbagEpisode],
        output_root: str,
        task: str,
        config: DictConfig,
        fps: int = 10,):

    repo_id = "kanazawanaoaki/my_test"

    features = {
        "observation.images.head": {
            "dtype": "video",
            "shape": list(config.obs.head_image.dim),
            "names": ["height", "width", "channel"],
        },
        "observation.images.second": {
            "dtype": "video",
            "shape": list(config.obs.second_image.dim),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (config.obs.robot_state.dim,),
            "names": {
                "motors": [
                    "r_shoulder_pan_joint",
                    "r_shoulder_lift_joint",
                    "r_upper_arm_roll_joint",
                    "r_elbow_flex_joint",
                    "r_forearm_roll_joint",
                    "r_wrist_flex_joint",
                    "r_wrist_roll_joint",
                    "r_gripper_joint"
                ]
            },
            "fps": config.ros.rate
        },
        "action": {
            "dtype": "float32",
            "shape": (config.actions.dim,),
            "names": {
                "motors": [
                    "r_shoulder_pan_joint",
                    "r_shoulder_lift_joint",
                    "r_upper_arm_roll_joint",
                    "r_elbow_flex_joint",
                    "r_forearm_roll_joint",
                    "r_wrist_flex_joint",
                    "r_wrist_roll_joint",
                    "r_gripper_joint"
                ]
            },
            "fps": config.ros.rate
        },
        # "timestamp":       Value("float32"),
        # "episode_index":   Value("int64"),
    }

    # import ipdb
    # ipdb.set_trace()

    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="pr2",
        root=output_root,
        fps=int(fps),
        use_videos=True,
        features=features,
        image_writer_threads=5,
        image_writer_processes=10,
    )

    ## Add frames
    for idx, ep in enumerate(tqdm(episode_list, desc="Episodes")):
        # import ipdb
        # ipdb.set_trace()
        for i in range(len(ep.head_images)):
            image_dict = {
                "observation.images.head": ep.head_images[i],
                "observation.images.second": ep.second_images[i],
            }
            lerobot_dataset.add_frame(
                {
                    **image_dict,
                    "observation.state": ep.states[i],
                    "action": ep.actions[i],
                },
                task=task,
            )
        lerobot_dataset.save_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str)
    args = parser.parse_args()

    current_time = datetime.datetime.now()
    print("[program start time] :", current_time.strftime("%Y-%m-%d %H:%M:%S"))

    config = FileUtils.get_config_from_project_name(args.project_name)

    data_path = os.path.join(FileUtils.get_data_dir(args.project_name), "rosbag_episode.pkl")
    with open(data_path, 'rb') as file:
        episode_list = pickle.load(file)

    base_data_dir = Path(FileUtils.get_data_dir(args.project_name))
    output_root   = base_data_dir / 'lerobot_data'
    convert_to_lerobot_v21(episode_list=episode_list, output_root=output_root, task=config.task['language_instruction'], config=config, fps=config.ros.rate)
