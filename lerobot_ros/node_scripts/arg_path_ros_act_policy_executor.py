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

import imitator.utils.file_utils as FileUtils

# from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, JointState
from eus_imitation_msgs.msg import FloatVector

class InferenceNode(object):
    def __init__(self, cfg: Dict[str, Any], project_name: str, model_path: str, hz=3):
         # ノードの初期化
        rospy.init_node('act_policy_executor', anonymous=True)

        self.current_head_image = None
        self.current_second_image = None
        self.current_robot_state = None

        self.head_image_sub = rospy.Subscriber(cfg.obs.head_image.topic_name, CompressedImage, self.head_image_callback)
        self.second_image_sub = rospy.Subscriber(cfg.obs.second_image.topic_name, CompressedImage, self.second_image_callback)
        self.robot_state_sub = rospy.Subscriber(cfg.obs.robot_state.topic_name, FloatVector, self.robot_state_callback)

        self.robot_action_pub = rospy.Publisher(cfg.actions.topic_name, FloatVector, queue_size=10)

        # CvBridgeインスタンスの作成
        self.bridge = CvBridge()

        self.config = cfg
        self.hz = hz

        # setup policy
        # model_dir = FileUtils.get_models_dir(project_name)
        # pretrained_policy_path = os.path.join(model_dir, "pretrained_model")
        pretrained_policy_path = model_path

        self.policy = ACTPolicy.from_pretrained(pretrained_policy_path)
        self.policy.to("cuda")
        self.policy.eval()

        ### set timer callback
        self.timer = rospy.Timer(rospy.Duration(1.0/hz), self.timer_callback)
        print("finish init!")

    def head_image_callback(self, msg):
        try:
            # ROSメッセージからOpenCV画像に変換
            # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8").astype(np.uinit8)
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            self.current_head_image = cv_image

        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def second_image_callback(self, msg):
        try:
            # ROSメッセージからOpenCV画像に変換
            # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8").astype(np.uinit8)
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

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
                "observation.images.head": head_image,
                "observation.images.second": second_image,
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
    parser.add_argument("--model_path", "-m", type=str, default="/media/ptolemy/73B2/kanazawa/datas/lerobot_output/train/pr2_grasp_green_bowl_20250708/checkpoints/100000/pretrained_model", help="path to pretrained model")
    parser.add_argument("-hz", type=int, default=3, help="ros node hz")
    args = parser.parse_args()

    project_name: str = args.pn
    hz: int = args.hz
    model_path: str = args.model_path

    config = FileUtils.get_config_from_project_name(project_name)

    inference_node = InferenceNode(config, project_name, model_path, hz)
    inference_node.spin()
