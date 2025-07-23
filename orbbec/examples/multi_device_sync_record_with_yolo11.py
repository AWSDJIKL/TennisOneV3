# ******************************************************************************
#  Copyright (c) 2023 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
'''
单个摄像头正常，2个会有明显加速，不推荐使用
'''
import json
import os
import time
import datetime
from queue import Queue
from threading import Lock
from typing import List

import cv2
import numpy as np
import open3d as o3d

from pyorbbecsdk import *
from utils import frame_to_bgr_image
from ultralytics import YOLO

frames_queue_lock = Lock()

# Configuration settings
MAX_DEVICES = 2
MAX_QUEUE_SIZE = 5
ESC_KEY = 27
# save_points_dir = os.path.join(os.getcwd(), "point_clouds")
# save_depth_image_dir = os.path.join(os.getcwd(), "depth_images")
save_color_image_dir = os.path.join(os.getcwd(), "color_images")

frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
stop_processing = False
curr_device_cnt = 0

# Load config file for multiple devices
config_file_path = os.path.join(os.path.dirname(__file__), "../config/multi_device_sync_config.json")
multi_device_sync_config = {}


def calculate_angle(a, b, c):
    # 输入：三个点的坐标 (x, y)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1, 1))  # 避免数值溢出
    return np.degrees(angle)


def get_yolo11_keypoints(frame):
    result = yolo11_model.predict(source=frame)[0]  # predict on an image
    keypoints = result.keypoints.xy[0].cpu().numpy()

    # 如果没有keypoints，说明没有检测到人，放弃这一帧
    if len(keypoints) == 0:
        # print("no keypoints detected, skip this frame")
        return []
    # 遍历keypoints，如果有关键点坐标为0，说明有一些点检测不到，放弃这一帧
    k_id = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    flag = False
    for i in k_id:
        if not keypoints[i][0] > 0 and not keypoints[i][1] > 0:
            print("keypoint {} not detected".format(i))
            flag = True
            break
    if flag:
        # print("have some keypoint not detected, skip this frame")
        return []
    return keypoints


def show2Dpose(kps, img):
    colors = [
        (255, 0, 0),  # 头到肩部
        (0, 255, 0),  # 左手
        (0, 0, 255),  # 右手
        (255, 255, 0),  # 身体
        (255, 0, 255),  # 左腿
        (0, 255, 255)  # 右腿
    ]

    connections = [[0, 5], [0, 6], [5, 6],
                   [5, 7], [7, 9],
                   [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [13, 15], [12, 14], [14, 16]]

    LR = [1, 1, 4, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6]

    thickness = 3

    for j, c in enumerate(connections):
        # print(kps.shape)
        if kps[c[0]][0] == 0 and kps[c[0]][1] == 0:
            continue
        if kps[c[1]][0] == 0 and kps[c[1]][1] == 0:
            continue
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j] - 1], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j] - 1], radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j] - 1], radius=3)

    # 定义需要计算角度的关节三元组
    angle_joints = [
        (5, 7, 9, "L_Elbow"),  # 左肘
        (6, 8, 10, "R_Elbow"),  # 右肘
        (11, 13, 15, "L_Knee"),  # 左膝
        (12, 14, 16, "R_Knee")  # 右膝
    ]

    for joint in angle_joints:
        a_idx, b_idx, c_idx, label = joint
        a = np.array(kps[a_idx][:2])
        b = np.array(kps[b_idx][:2])
        c = np.array(kps[c_idx][:2])
        if a[0] == 0 and a[1] == 0:
            continue
        if b[0] == 0 and b[1] == 0:
            continue
        if c[0] == 0 and c[1] == 0:
            continue
        # # 检查置信度（如果scores可用）
        # if scores is not None:
        #     if scores[a_idx] < conf_threshold or scores[b_idx] < conf_threshold or scores[c_idx] < conf_threshold:
        #         continue

        # 计算角度
        angle = calculate_angle(a, b, c)

        # 在图像上标注角度（以B点为中心）
        text_pos = (int(b[0]), int(b[1] - 10))  # 在关节点上方显示
        cv2.putText(
            img,
            f"{joint[-1]}{angle:.1f}",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),  # 蓝色字体
            2
        )

    return img


def convert_to_o3d_point_cloud(points, colors=None):
    """
    Converts numpy arrays of points and colors (if provided) into an Open3D point cloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Assuming colors are in [0, 255]
    return pcd


def read_config(config_file: str):
    global multi_device_sync_config
    with open(config_file, "r") as f:
        config = json.load(f)
    for device in config["devices"]:
        multi_device_sync_config[device["serial_number"]] = device
        print(f"Device {device['serial_number']}: {device['config']['mode']}")


def sync_mode_from_str(sync_mode_str: str) -> OBMultiDeviceSyncMode:
    sync_mode_str = sync_mode_str.upper()
    if sync_mode_str == "FREE_RUN":
        return OBMultiDeviceSyncMode.FREE_RUN
    elif sync_mode_str == "STANDALONE":
        return OBMultiDeviceSyncMode.STANDALONE
    elif sync_mode_str == "PRIMARY":
        return OBMultiDeviceSyncMode.PRIMARY
    elif sync_mode_str == "SECONDARY":
        return OBMultiDeviceSyncMode.SECONDARY
    elif sync_mode_str == "SECONDARY_SYNCED":
        return OBMultiDeviceSyncMode.SECONDARY_SYNCED
    elif sync_mode_str == "SOFTWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.SOFTWARE_TRIGGERING
    elif sync_mode_str == "HARDWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.HARDWARE_TRIGGERING
    else:
        raise ValueError(f"Invalid sync mode: {sync_mode_str}")


# Frame processing and saving
def process_frames(pipelines, video_writers: List[cv2.VideoWriter]):
    global frames_queue
    global stop_processing
    global curr_device_cnt, save_points_dir, save_depth_image_dir, save_color_image_dir
    while not stop_processing:
        now = time.time()
        for device_index in range(curr_device_cnt):
            with frames_queue_lock:
                frames = frames_queue[device_index].get() if not frames_queue[device_index].empty() else None
            if frames is None:
                # print(f"Device {device_index} has no frames")
                continue
            color_frame = frames.get_color_frame() if frames else None
            # depth_frame = frames.get_depth_frame() if frames else None
            pipeline = pipelines[device_index]
            video_writer = video_writers[device_index]
            # print(f"Device {device_index} frames")

            if color_frame:
                color_image = frame_to_bgr_image(color_frame)
                yolo_keypoints = get_yolo11_keypoints(color_image)
                if len(yolo_keypoints) > 0:
                    print("getting keypoints")
                    color_image = show2Dpose(yolo_keypoints, color_image)
                # 在左上角标注当前时间
                # 获取当前时间并格式化
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # 将当前时间打印在帧的左上角
                # 参数分别是：图像、文字、位置、字体、字体大小、颜色、厚度
                color_image = cv2.putText(color_image, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                          1, (255, 255, 255), 2, cv2.LINE_AA)
                video_writer.write(color_image)
                print(f"Device {device_index}  Processing time: {time.time() - now:.3f}s")
                # print("saved color image")
                # color_filename = os.path.join(save_color_image_dir,
                #                               f"color_{device_index}_{color_frame.get_timestamp()}.png")
                # cv2.imwrite(color_filename, color_image)

            # if depth_frame:
            #     timestamp = depth_frame.get_timestamp()
            #     width = depth_frame.get_width()
            #     height = depth_frame.get_height()
            #     timestamp = depth_frame.get_timestamp()
            #     scale = depth_frame.get_depth_scale()
            #     data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            #     data = data.reshape((height, width))
            #     data = data.astype(np.float32) * scale
            #     data = data.astype(np.uint16)
            #     if not os.path.exists(save_depth_image_dir):
            #         os.mkdir(save_depth_image_dir)
            #     raw_filename = save_depth_image_dir + "/depth_{}x{}_device_{}_{}.raw".format(width, height,
            #                                                                                  device_index, timestamp)
            #     data.tofile(raw_filename)
            #     camera_param = pipeline.get_camera_param()
            #     points = frames.get_point_cloud(camera_param)
            #     if len(points) == 0:
            #         print("no depth points")
            #         continue
            #     points_array = np.array([tuple(point) for point in points],
            #                             dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            #     if not os.path.exists(save_points_dir):
            #         os.mkdir(save_points_dir)
            #     points_filename = os.path.join(save_points_dir, f"points_device_{device_index}_{timestamp}.ply")
            #     pcd = convert_to_o3d_point_cloud(np.array(points))
            #     o3d.io.write_point_cloud(points_filename, pcd)
        # print(f"Processing time: {time.time() - now:.3f}s")


def on_new_frame_callback(frames: FrameSet, index: int):
    global frames_queue
    global MAX_QUEUE_SIZE
    assert index < MAX_DEVICES
    with frames_queue_lock:
        if frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            frames_queue[index].get()
        frames_queue[index].put(frames)


def start_streams(pipelines: List[Pipeline], configs: List[Config]):
    index = 0
    for pipeline, config in zip(pipelines, configs):
        print(f"Starting device {index}")
        pipeline.start(
            config,
            lambda frame_set, curr_index=index: on_new_frame_callback(
                frame_set, curr_index
            ),
        )
        pipeline.enable_frame_sync()
        index += 1


def stop_streams(pipelines: List[Pipeline], video_writers: List[cv2.VideoWriter]):
    index = 0
    for pipeline in pipelines:
        print(f"Stopping device {index}")
        pipeline.stop()
        index += 1
    for video_writer in video_writers:
        video_writer.release()


# Main function for setup and teardown
def main():
    global curr_device_cnt
    read_config(config_file_path)
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("No device connected")
        return
    pipelines = []
    configs = []
    video_writers = []
    curr_device_cnt = device_list.get_count()
    for i in range(min(device_list.get_count(), MAX_DEVICES)):
        device = device_list.get_device_by_index(i)
        pipeline = Pipeline(device)
        config = Config()
        serial_number = device.get_device_info().get_serial_number()
        sync_config_json = multi_device_sync_config[serial_number]
        sync_config = device.get_multi_device_sync_config()
        sync_config.mode = sync_mode_from_str(sync_config_json["config"]["mode"])
        sync_config.color_delay_us = sync_config_json["config"]["color_delay_us"]
        sync_config.depth_delay_us = sync_config_json["config"]["depth_delay_us"]
        sync_config.trigger_out_enable = sync_config_json["config"]["trigger_out_enable"]
        sync_config.trigger_out_delay_us = sync_config_json["config"]["trigger_out_delay_us"]
        sync_config.frames_per_trigger = sync_config_json["config"]["frames_per_trigger"]
        device.set_multi_device_sync_config(sync_config)
        print(f"Device {serial_number} sync config: {sync_config}")

        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

        color_profile = profile_list.get_default_video_stream_profile()
        # color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB, 30)
        config.enable_stream(color_profile)
        print(f"Device {serial_number} color profile: {color_profile}")

        # profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        # depth_profile = profile_list.get_default_video_stream_profile()
        # print(f"Device {serial_number} depth profile: {depth_profile}")
        # config.enable_stream(depth_profile)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        folder_path = "../../video"
        # 判断文件夹是否存在
        if not os.path.exists(folder_path):
            # 不存在则创建
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 已创建")
        else:
            print(f"文件夹 '{folder_path}' 已存在")
        video_writer = cv2.VideoWriter(f"../../video/yolo11_{serial_number}_1920_1080_30_{current_time}.mp4", fourcc,
                                       30,
                                       (1920, 1080))
        video_writers.append(video_writer)
        pipelines.append(pipeline)
        configs.append(config)
    start_streams(pipelines, configs)
    global stop_processing
    try:
        process_frames(pipelines, video_writers)
    except KeyboardInterrupt:
        print("Interrupted by user")
        stop_processing = True
    finally:
        print("===============Stopping pipelines====")
        stop_streams(pipelines, video_writers)


if __name__ == "__main__":
    yolo11_model = YOLO("yolo11x-pose.pt", "pose")
    main()
