import os
import pickle
from pathlib import Path
from imitator.utils.file_utils import get_data_dir, get_config_from_project_name
import argparse

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

# from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig

def check_episode_data(project_name: str):
    # 設定とデータの読み込み
    config = get_config_from_project_name(project_name)
    data_path = os.path.join(get_data_dir(project_name), "rosbag_episode.pkl")

    with open(data_path, "rb") as f:
        episode_list = pickle.load(f)

    print(f"\n✅ Loaded {len(episode_list)} episodes\n")

    total_episodes = len(episode_list)
    valid_episodes = 0

    for ep_idx, ep in enumerate(episode_list):
        print(f"--- Checking Episode {ep_idx} ---")

        head_images = ep.head_images
        second_images = ep.second_images
        states = ep.states
        actions = ep.actions

        # データ長チェック
        lens = list(map(len, [head_images, second_images, states, actions]))
        if len(set(lens)) != 1:
            print(f"❌ Length mismatch: head={len(head_images)}, second={len(second_images)}, states={len(states)}, actions={len(actions)}")
            continue

        if lens[0] == 0:
            print(f"⚠️  No frames in episode {ep_idx}")
            continue

        # 先頭フレームのshapeチェック
        try:
            hi_shape = head_images[0].shape
            si_shape = second_images[0].shape
            st_shape = states[0].shape
            ac_shape = actions[0].shape

            print(f"✅ Frame shapes: head={hi_shape}, second={si_shape}, state={st_shape}, action={ac_shape}")

            # None チェック
            for i in range(lens[0]):
                if (
                    head_images[i] is None or
                    second_images[i] is None or
                    states[i] is None or
                    actions[i] is None
                ):
                    print(f"❌ None data found at frame {i}")
                    break
            else:
                valid_episodes += 1
                print("✅ Episode is valid.")

        except Exception as e:
            print(f"❌ Error reading frame data: {e}")

    print(f"\n✅ Summary: {valid_episodes}/{total_episodes} episodes valid.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", "--project_name", type=str, required=True)
    args = parser.parse_args()
    check_episode_data(args.project_name)
