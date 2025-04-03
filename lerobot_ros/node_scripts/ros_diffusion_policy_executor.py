#!/usr/bin/env python
#-*- coding: utf-8 -*-

import argparse
import cv2
import glob
import numpy as np
import os
from pathlib import Path
import pickle
import socket
import struct
import torch
from typing import Any, Dict, List, Optional, Type

from imitator.utils.file_utils import (get_config_from_project_name,
                                       get_models_dir)

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionConfig, DiffusionPolicy

from safetensors.torch import load_file

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, JointState
from eus_imitation_msgs.msg import FloatVector

class InferenceNode(object):
    def __init__(self, cfg: Dict[str, Any], project_name: str, n_pixel=112):
         # ノードの初期化
        rospy.init_node('diffusion_policy_executor', anonymous=True)

        self.current_head_image = None
        self.current_second_image = None
        self.current_robot_state = None

        self.head_image_sub = rospy.Subscriber(cfg.obs.head_image.topic_name, CompressedImage, self.head_image_callback)
        self.second_image_sub = rospy.Subscriber(cfg.obs.second_image.topic_name, CompressedImage, self.second_image_callback)
        self.robot_state_sub = rospy.Subscriber(cfg.obs.robot_state.topic_name, FloatVector, self.robot_state_callback)

        self.robot_action_pub = rospy.Publisher(cfg.actions.topic_name, FloatVector, queue_size=10)

        # CvBridgeインスタンスの作成
        self.bridge = CvBridge()

        self.n_pixel = n_pixel
        self.config = cfg

        # setup policy
        model_dir = FileUtils.get_models_dir(project_name)
        stats_path = os.path.join(FileUtils.get_data_dir(project_name), "stats.pkl")
        with Path(stats_path).open("rb") as f:
            stats = pickle.load(f)
        for key, value in stats.items():
            for key_sub, value_sub in value.items():
                stats[key][key_sub] = value_sub.to("cuda")
        print(stats)

        resol = self.n_pixel
        camera_names = ["head", "second"]
        input_shapes = {"observation.state": [config.obs.robot_state.dim]}
        for name in camera_names:
            input_shapes[f"observation.image.{name}"] = [3, resol, resol]
        output_shapes = {"action": [config.actions.dim]}
        normalization_mode = {"observation.state": "min_max"}
        for name in camera_names:
            normalization_mode[f"observation.image.{name}"] = "mean_std"
        cfg = DiffusionConfig(use_separate_rgb_encoder_per_camera=True, input_shapes=input_shapes, output_shapes=output_shapes, input_normalization_modes=normalization_mode)
        # cfg = DiffusionConfig(use_separate_rgb_encoder_per_camera=True)
        effective_keys = list(cfg.output_shapes.keys()) + list(cfg.input_shapes.keys()) + ["episode_index", "frame_indx", "index", "next.done",  "timestamp"]
        effective_key_set = set(effective_keys)
        for key, value in stats.items():
            if key not in effective_key_set:
                continue
            inner_dict = {}
            for key_inner, value_inner in value.items():
                inner_dict[key_inner] = torch.tensor(value_inner)
        stats[key] = inner_dict

        policy = DiffusionPolicy(cfg, dataset_stats=stats)
        self.policy = policy.to("cuda")
        pretrained_weights = load_file(os.path.join(model_dir, "model.safetensors"))
        self.policy.load_state_dict(pretrained_weights)


        ### set timer callback
        # self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)
        # self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        self.timer = rospy.Timer(rospy.Duration(0.33), self.timer_callback) ## これは結構動く with 400
        # self.timer = rospy.Timer(rospy.Duration(0.25), self.timer_callback) ## 良かった
        # self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback)
        # self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        # self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_callback)
        print("finish init!")

    def head_image_callback(self, msg):
        try:
            # ROSメッセージからOpenCV画像に変換
            # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8").astype(np.uinit8)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            self.current_head_image = cv_image

        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def second_image_callback(self, msg):
        try:
            # ROSメッセージからOpenCV画像に変換
            # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8").astype(np.uinit8)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            self.current_second_image = cv_image

        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def robot_state_callback(self, msg):
        try:
            # ROSメッセージからOpenCV画像に変換
            # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            robot_state = msg.data

            self.current_robot_state = robot_state

        except Exception as e:
            rospy.logerr(f"Error converting robot_state: {e}")

    def timer_callback(self, event):
        if (self.current_head_image is not None) and (self.current_second_image is not None) and (self.current_robot_state is not None):
            print(self.current_head_image is not None, self.current_second_image is not None, self.current_robot_state is not None)
            # print(self.current_robot_state)
            head_rgb = cv2.resize(self.current_head_image, tuple(self.config['obs']['head_image'].dim[:2]), interpolation=cv2.INTER_LANCZOS4)
            second_rgb = cv2.resize(self.current_second_image, tuple(self.config['obs']['head_image'].dim[:2]), interpolation=cv2.INTER_LANCZOS4)
            head_image = torch.from_numpy(head_rgb).float()
            second_image = torch.from_numpy(second_rgb).float()
            head_image = head_image.to(torch.float32) / 255
            second_image = second_image.to(torch.float32) / 255
            head_image = head_image.permute(2, 0, 1)
            second_image = second_image.permute(2, 0, 1)
            head_image = head_image.unsqueeze(0)
            second_image = second_image.unsqueeze(0)
            # print(self.current_robot_state)

            state = torch.from_numpy(np.array(self.current_robot_state)).float().unsqueeze(0)
            # state = torch.from_numpy(np.array(data['float_list']).float().unsqueeze(0)
            head_image = head_image.to("cuda")
            second_image = second_image.to("cuda")
            state = state.to("cuda")
            # print(state)

            observation = {
                # "observation.image": head_image,
                "observation.image.head": head_image,
                "observation.image.second": second_image,
                "observation.state": state
            }
            with torch.inference_mode():
                action = self.policy.select_action(observation)
            action_np = action.detach().cpu().numpy().flatten()
            print(action_np)

            action_msg = FloatVector()
            action_msg.header.stamp = rospy.Time.now()
            action_msg.data = action_np
            self.robot_action_pub.publish(action_msg)

        else:
            print(self.current_head_image is not None, self.current_second_image is not None, self.current_robot_state is not None)


    def spin(self):
        # ノードを実行し続ける
        rospy.spin()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pn", type=str, default="wrapping", help="project name")
    parser.add_argument("-m", type=int, default=112, help="pixel num")
    args = parser.parse_args()

    n_pixel: int = args.m
    project_name: str = args.pn

    config = get_config_from_project_name(project_name)

    inference_node = InferenceNode(config, project_name, n_pixel)
    inference_node.spin()
