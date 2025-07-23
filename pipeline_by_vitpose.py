import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import yaml
import os
import argparse
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
# from ultralytics import YOLO
from skopt import gp_minimize
import torch
import random
from torch.utils.data import DataLoader
from torchlight import DictAction
from TrackNetV3.test import predict_location, get_ensemble_weight, generate_inpaint_mask
from TrackNetV3.dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from TrackNetV3.utils.general import *
import sys
# from patsy import desc

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
# params["model_folder"] = "../../models/"
params["model_folder"] = "/home/awsdjikl/TrackNetV3/models"
params["maximize_positives"] = "True"
# opWrapper = op.WrapperPython()
# opWrapper.configure(params)
# opWrapper.start()

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
from PIL import Image
from TrackNetV3.GCN.main import Processor, get_parser
from TrackNetV3.GCN import graph

CATEGORIES = ["forehand_backswing", "backhand_backswing", "backhand_power_stroke", "forehand_power_stroke", "forehand_follow_through", "backhand_follow_through", "others"]
WINDOW_SIZE = 30

person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-base")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-base", device_map=device)


def calculate_angle(a, b, c):
    # 输入：三个点的坐标 (x, y)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1, 1))  # 避免数值溢出
    return np.degrees(angle)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    # print(f'Batch size: {batch_size}, Sequence length: {seq_len}')
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    vis = None
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        vis = y_pred.flatten(start_dim=2).any(dim=2)
        # vis = 1 if torch.any(y_pred).item() else 0  # Visibility is 1 if any pixel in heatmap is above threshold, else 0
        # print("-"* 20)
        # print(vis)
        # print(vis) 
        # print(y_pred)
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)  # (N, L, H, W)
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    if vis is not None and vis[n][f]:
                        cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT * img_scaler[1])
                    else:
                        # print(f'Frame {f_i} has no visible object, skipping...')
                        cx_pred, cy_pred = 0, 0
                    # cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT * img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    if vis[n][f]:
                        cx_pred, cy_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(bbox_pred[1] + bbox_pred[3] / 2)
                        cx_pred, cy_pred = int(cx_pred * img_scaler[0]), int(cy_pred * img_scaler[1])
                    else:
                        # print(f'Frame {f_i} has no visible object, skipping...')
                        cx_pred, cy_pred = 0, 0
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                # print(f'Frame {f_i}, X: {cx_pred}, Y: {cy_pred}, Visibility: {vis_pred}')
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    


def get_shot_frame(ball_data):
    # 取x轴坐标，计算delta_x
    ball_data = ball_data[["Frame", 'X']]
    # 初始化变量
    segments = []
    current_segment = []

    # 按0分段
    for index, row in ball_data.iterrows():
        if row['X'] == 0:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(row)

    if current_segment:
        segments.append(current_segment)
    # 处理每个分段
    all_strike_positions = []
    for segment in segments:
        if len(segment) > 5:
            segment_df = pd.DataFrame(segment)
            strike_positions = find_strike_positions(segment_df)
            all_strike_positions.extend(strike_positions)
    print("球被击打的位置:", all_strike_positions)

    return all_strike_positions
    

# 函数用于计算梯度变化的位置
def find_strike_positions(segment_df):
    segment_df['Gradient'] = segment_df['X'].diff()
    start_index = segment_df.index[0]
    strike_positions = []
    for i in range(start_index + 1, start_index + len(segment_df)):
        if (segment_df.loc[i, 'Gradient'] > 0 and segment_df.loc[i - 1, 'Gradient'] < 0) or \
           (segment_df.loc[i, 'Gradient'] < 0 and segment_df.loc[i - 1, 'Gradient'] > 0):
            strike_positions.append(segment_df.loc[i, 'Frame'])
    return strike_positions


def plot_ball_frame(ball_frame):
    x = ball_frame['X'].values
    plt.figure(figsize=(10, 5))
    plt.plot(x, marker='o', linestyle='-', color='b', label='X Coordinate')
    plt.savefig('/home/awsdjikl/TrackNetV3/prediction/game3_ball_all.png')


def slice_videos(video_path, all_strike_positions, lenth=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {frame_count}, 帧率: {fps}, 分辨率: {width}x{height}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = Path(video_path)
    video_name = video_path.stem
    output_dir = video_path.parent / Path(video_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_list = []
    current_frame = -1
    for strike_position in all_strike_positions:
        start_frame = max(0, strike_position - int(lenth / 2))
        end_frame = min(frame_count, strike_position + int(lenth / 2) - 1)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        output_path = output_dir / Path(f"{video_name}_{start_frame}_{end_frame}_{strike_position}.mp4")
        video_list.append(output_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame - 1
        while True:
            ret, frame = cap.read()
            current_frame += 1
            # print(current_frame)
            if not ret:
                break
            if current_frame >= start_frame and current_frame <= end_frame:
                # print(current_frame)
                out.write(frame)
            if current_frame > end_frame:
                break
        out.release()
    cap.release()
    return video_list


def show_vitpose_pose(kp_list, img_list, video_writer, cls=None, ball_position=None):
    colors = [
        (255, 0, 0),  # 头到肩部
        (0, 255, 0),  # 左手
        (0, 0, 255),  # 右手
        (255, 255, 0),  # 身体
        (255, 0, 255),  # 左腿
        (0, 255, 255)  # 右腿
    ]
    angle_joints = [
    (5, 7, 9, "L Elbow", (114, 255, 250)),  # 左肘
    (6, 8, 10, "R Elbow", (231, 237, 62)),  # 右肘
    (11, 13, 15, "L Knee", (0, 117, 255)),  # 左膝
    (12, 14, 16, "R Knee", (193, 182, 255))  # 右膝
]
    suggest_angle_range = [
    (90, 170),  # 左肘
    (90, 170),  # 右肘
    (150, 175),  # 左膝
    (150, 175)  # 右膝
]
    joint_color = [
    # BGR
    (114, 255, 250),  # 左手
    (231, 237, 62),  # 右手
    (0, 117, 255),  # 左腿
    (193, 182, 255)  # 右腿
]

    # openpose
    connections = [[0, 1], [0, 2], [0, 5], [0, 6],
                [5, 7], [7, 9],
                [6, 8], [8, 10],
                [5, 6], [5, 11], [6, 12], [11, 12],
                [11, 13], [13, 15],
                [12, 14], [14, 16]]
    LR = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6]

    thickness = 3

    # 定义需要计算角度的关节三元组
    all_angles = []
    # all_frames = []
    # 读取每一帧
    print("kps len:", len(kp_list))
    # with tqdm(desc="calculate angle", unit="帧", total=len(kp_list)) as pbar:
    for kps in tqdm(kp_list, desc="calculate angle", unit="帧"):
            # 计算这一帧的角度
            angles = []
            for i, joint in enumerate(angle_joints):
                a_idx, b_idx, c_idx, label, text_color = joint
                if max(a_idx, b_idx, c_idx) >= len(kps):
                    print(f"Warning: Joint indices {a_idx}, {b_idx}, {c_idx} exceed keypoints length {len(kps)}")
                    angles.append(0)  # 或者处理为其他默认值
                    continue
                a = np.array(kps[a_idx][:2])
                b = np.array(kps[b_idx][:2])
                c = np.array(kps[c_idx][:2])
                # if a[0] == 0 and a[1] == 0:
                #     continue
                # if b[0] == 0 and b[1] == 0:
                #     continue
                # if c[0] == 0 and c[1] == 0:
                #     continue
                # # 检查置信度（如果scores可用）
                # if scores is not None:
                #     if scores[a_idx] < conf_threshold or scores[b_idx] < conf_threshold or scores[c_idx] < conf_threshold:
                #         continue

                # 计算角度
                angle = calculate_angle(a, b, c).tolist()
                angles.append(angle)
            all_angles.append(angles)
            # all_frames.append(img)
    # 初始化超出范围的计数器
    out_of_range_counts = np.zeros(len(angle_joints))
    for angles in all_angles:
        for i, (min_angle, max_angle) in enumerate(suggest_angle_range):
            if angles[i] < min_angle or angles[i] > max_angle:
                out_of_range_counts[i] += 1

    # 计算百分比
    out_of_range_percent = out_of_range_counts / len(all_angles)
    rect_width = 600
    rect_height = 400
    # rect_height = 400
    # if display_in_video:
    #     rect_x = width - rect_width
    #     rect_height = 400
    # else:
    #     rect_x = width
    #     rect_height = height
    width = img_list[0].shape[1]
    height = img_list[0].shape[0]
    rect_x = width - rect_width
    rect_y = 0
    # 定义半透明矩形的颜色和透明度
    color_rect = (0, 0, 0)
    alpha = 0.75  # 透明度
    # 定义圆环的位置、半径和完整度
    circle_centers = [
        (rect_x + 75, rect_y + 180),
        (rect_x + 225, rect_y + 180),
        (rect_x + 375, rect_y + 180),
        (rect_x + 525, rect_y + 180)
    ]
    circle_radii = [(40, 40)] * 4  # 半长轴和半短轴相同，形成圆形
    circle_thickness = 10
    circle_completion_factors = [1 - p for p in out_of_range_percent]  # 圆环的完成度
    user_picture = cv2.imread("TrackNetV3/user/tai.jpg")
    user_picture = cv2.resize(user_picture, (80, 80))
    # 创建一个空白图像作为掩膜
    mask = np.zeros((80, 80), dtype=np.uint8)
    # 在掩膜上绘制一个白色填充的圆
    circle_mask = cv2.circle(mask, (40, 40), 40, 255, -1)
    # 将原图与掩膜结合
    user_picture = cv2.bitwise_and(user_picture, user_picture, mask=circle_mask)
    icon = cv2.imread("TrackNetV3/user/icon.png")
    icon = cv2.resize(icon, (40, 40))
    # 创建一个空白图像作为掩膜
    icon_mask = np.zeros((40, 40), dtype=np.uint8)
    # 在掩膜上绘制一个白色填充的圆
    icon_circle_mask = cv2.circle(icon_mask, (20, 20), 20, 255, -1)
    icon = cv2.bitwise_and(icon, icon, mask=icon_circle_mask)
    for kps, img, ball_data in tzip(kp_list, img_list, ball_position, desc="绘制关节和角度", unit="帧"):
        for j, c in enumerate(connections):
            # print(kps.shape)
            if max(c) >= len(kps):
                # print(f"Warning: Connection indices {c} exceed keypoints length {len(kps)}")
                continue
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
        if ball_data:
            # 在图像上绘制球的位置
            ball_x = ball_data[0]
            ball_y = ball_data[1]
            if ball_x != 0 and ball_y != 0:
                cv2.circle(img, (int(ball_x), int(ball_y)), radius=5, color=(0, 255, 255), thickness=-1)
                cv2.putText(img, 'Ball', (int(ball_x) + 10, int(ball_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        overlay = img.copy()
        # 在透明图层上绘制矩形
        cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), color_rect, -1)
        # print(angles)
        # 左上角画用户头像
        up_h, up_w = user_picture.shape[:2]
        # print(overlay.shape)
        # print(user_picture.shape)
        # print(rect_x + 30, rect_x + 30 + up_h)
        # print(overlay[rect_y + 15:rect_y + 15 + up_w, rect_x + 15:rect_x + 15 + up_h,:].shape)
        # print(overlay[rect_x + 15:rect_x + 15 + up_h, rect_y + 15:rect_y + 15 + up_w,:].shape)
        overlay[rect_y + 15:rect_y + 15 + up_w, rect_x + 15:rect_x + 15 + up_h,:] = user_picture
        # 头像下方写当前的动作
        cv2.putText(
            overlay,
            f"Action: {cls}", (rect_x + 15, rect_y + 40 + up_w), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 1
        )
        # 右边画用户信息
        cv2.putText(
            overlay,
            f"User Name: Tai", (rect_x + 30 + up_h, rect_y + 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 1
        )
        cv2.putText(
            overlay,
            f"Dominant hand: Right", (rect_x + 30 + up_h, rect_y + 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 1
        )
        # 右上角绘制Tennis One
        ic_h, ic_w = icon.shape[:2]
        overlay[rect_y + 15:rect_y + 15 + ic_h, rect_x + 400:rect_x + 400 + ic_w,:] = icon
        cv2.putText(
            overlay,
            f"TENNIS ONE", (rect_x + 450, rect_y + 45), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 1
        )
        # 绘制不完整的圆环
        for center, radius, color, completion_factor, angle_range, out_p, angle, (_, _, _, joint_name,
                                                                                _) in zip(
            circle_centers,
            circle_radii,
            joint_color,
            circle_completion_factors,
            suggest_angle_range,
            out_of_range_percent,
            angles, angle_joints):
            start_angle = 0
            end_angle = int(completion_factor * 360)
            cv2.ellipse(overlay, center, radius, 0, start_angle, end_angle, color, circle_thickness)
            # 在圆心写上分数
            score = 100 * math.log(completion_factor + 1) / math.log(2)
            cv2.putText(
                overlay,
                f"{score :.1f}", (center[0] - int(radius[0] / 2) - 5, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 1
            )
            # 在圆底下写
            # 关节名称
            # 推荐角度范围
            # x%的时间大于范围
            # x%的时间小于范围
            # 当前帧的角度
            cv2.putText(
                overlay,
                # f"Suggest angle range:({angle_range[0]},{angle_range[1]})",
                f"{joint_name}",
                (center[0] - radius[0], center[1] + radius[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1
            )
            cv2.putText(
                overlay,
                # f"Suggest angle range:({angle_range[0]},{angle_range[1]})",
                f"({angle_range[0]},{angle_range[1]})",
                (center[0] - radius[0], center[1] + radius[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1
            )
            cv2.putText(
                overlay,
                f"{out_p * 100:.1f}% out", (center[0] - radius[0], center[1] + radius[1] + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                color, 1
            )
            cv2.putText(
                overlay,
                f"{(1 - out_p) * 100:.1f}% in", (center[0] - radius[0], center[1] + radius[1] + 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 1
            )
            cv2.putText(
                overlay,
                f"{angle:.1f}", (center[0] - radius[0], center[1] + radius[1] + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                color, 1
            )
            # print(angle)
        # 将透明图层与原始帧混合
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        video_writer.write(img)
    return


def get_keypoint(frame, last_position):
    '''
    frame: PIL.Image
    last_position: 上一帧球员的关节点位置或者球的位置 x, y
    '''
    # 预测
    # 先找人物框
    inputs = person_image_processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = person_model(**inputs)
    width = frame.shape[1]
    height = frame.shape[0]
    results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(height, width)]), threshold=0.3)
    result = results[0]
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()
    # # 选取所有人物框中距离last_position最近的人
    # if len(person_boxes) > 1:
    #     centers = (person_boxes[:, 0:2] + person_boxes[:, 2:4]) / 2.0
    #     distances = np.linalg.norm(centers - last_position, axis=1)
    #     closest_idx = np.argmin(distances)
    #     last_position = centers[closest_idx]
    #     person_box = person_boxes[closest_idx].reshape(1, 4)
    # elif len(person_boxes) == 0:
    #     last_position = (person_boxes[0, 0:2] + person_boxes[0, 2:4]) / 2.0
    #     person_box = person_boxes[0].reshape(1, 4)
    # else:
    #     print("no keypoints detected, skip this frame")
    #     return None, last_position
    if len(person_boxes) == 0:
        print("no keypoints detected, skip this frame")
        return None, last_position
    # 根据人物框提取骨骼
    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
    inputs = image_processor(frame, boxes=[person_boxes], return_tensors="pt").to(device)
    inputs["dataset_index"] = torch.tensor([0], device=device)
    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs)
    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes], threshold=0.3)
    result = pose_results[0][0]["keypoints"].cpu().numpy()  # results for first image
    return result, last_position


def find_nearest_person(joint_data, target_point):
    """
    joint_data: numpy数组, 形状为(n, 25, 3), 表示n个人的关节点数据
    target_point: 目标坐标点 [x, y]
    返回: 最近的人的索引
    """
    # 直接计算每个人的坐标算术平均（忽略置信度）
    centers = np.mean(joint_data[:,:,:2], axis=1)  # 形状 (n, 2)
    
    # 计算到目标点的欧氏距离
    distances = np.linalg.norm(centers - target_point, axis=1)
    
    # 返回最近的人的索引
    return np.argmin(distances)


def get_vitpose_keypoint(video_path:Path, ball_data:pd.DataFrame, model_list):
    # 先提取击球帧的人，若检测到多于一个人则取里球最近的人作为球员
    _, start_frame, end_frame, strike_position = video_path.stem.split('_')
    start_frame, end_frame , strike_position = int(start_frame), int(end_frame), int(strike_position)
    strike_position_in_video = strike_position - start_frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, strike_position_in_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ret, frame = cap.read()
    keypoint_list = []
    frame_list = []
    ball_position = []
    
    ball_strike_position = np.array([ball_data.loc[ball_data['Frame'] == strike_position, 'X'].values[0], ball_data.loc[ball_data['Frame'] == strike_position, 'Y'].values[0]])
    ball_position.append(ball_strike_position.tolist())
    keypoint, lastposition = get_keypoint(frame, ball_strike_position)
    if keypoint is None:
        print(f"No keypoint detected in frame {strike_position_in_video} of video {video_path.name}, skipping...")
        return None
    keypoint_list.append(keypoint)
    frame_list.append(frame)
    # 从击球帧开始逐帧向前读取，获取球员的关节点
    j = 1
    for i in range(strike_position_in_video - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        keypoint, lastposition = get_keypoint(frame, lastposition)
        if keypoint is None:
            print(f"No keypoint detected in frame {i} of video {video_path.name}, skipping...")
            continue
        keypoint_list.insert(0, keypoint)
        frame_list.insert(0, frame)
        ball_position.insert(0, [ball_data.loc[ball_data['Frame'] == strike_position - j, 'X'].values[0], ball_data.loc[ball_data['Frame'] == strike_position - j, 'Y'].values[0]])
        j += 1
    lastposition = ball_strike_position
    j = 1
    # 从击球帧开始逐帧向后读取，获取球员的关节点
    for i in range(strike_position_in_video + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        keypoint, lastposition = get_keypoint(frame, lastposition)
        if keypoint is None:
            print(f"No keypoint detected in frame {i} of video {video_path.name}, skipping...")
            continue
        keypoint_list.append(keypoint)
        frame_list.append(frame)
        ball_position.append([ball_data.loc[ball_data['Frame'] == strike_position + j, 'X'].values[0], ball_data.loc[ball_data['Frame'] == strike_position + j, 'Y'].values[0]])
        j += 1
    # keypoints = np.array(keypoint_list)[:,:17,:2]  # 只取x,y坐标
    # keypoints = keypoints.reshape(keypoints.shape[0], -1)
    # keypoints = np.array(keypoint_list)
    # print(f"Keypoints shape: {keypoints.shape}")
    # data_numpy = np.zeros((3, WINDOW_SIZE, 17, 2))
    # data_numpy = np.zeros((WINDOW_SIZE, 1, 17, 2))
    # keypoints = np.array(keypoint_list)[:,:17,:2]  # WINDOW_SIZE, 17, 2
    # print(f"Keypoints shape: {keypoints.shape}")
    # for i in range(len(keypoints)):
    #     data_numpy[i, 0,:,:] = keypoints[i]
    # input = torch.FloatTensor(data_numpy).permute(3, 0, 2, 1).cuda().unsqueeze(0) 
    # action_cls = CATEGORIES[get_cls(input, model_list, model_weight_list)]
    action_cls = CATEGORIES[-1]
    print(f"Action class for {video.name}: {action_cls}")

    save_name = video_path.parent / Path(video_path.stem + '_with_keypoint.mp4')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (width, height))
    # show_openpose_pose(keypoint_list, frame_list, video_writer, cls=action_cls, ball_position=ball_position)
    show_vitpose_pose(keypoint_list, frame_list, video_writer, cls=action_cls, ball_position=ball_position)

    # for kp, img , bp in zip(keypoint_list, frame_list, ball_position):
        # img = show_openpose_pose(kp, img, ball_data=bp)
        # img = plot_action_cls(action_cls, img)
        # video_writer.write(img)
    video_writer.release()
    cap.release()
    return keypoint_list, frame_list


def show_all_openpose_pose(video_path:Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_name = video_path.parent / Path(video_path.stem + '_with_keypoint.mp4')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoint = get_keypoint(frame)
        if keypoint is None:
            print("no keypoints detected, skip this frame")
            continue
        for i in range(keypoint.shape[0]):
            frame = show_openpose_pose(keypoint[i], frame)
        video_writer.write(frame)
    video_writer.release()
    cap.release()


def get_cls(input, model_list, model_weight_list):
    result = []
    for model, weight in zip(model_list, model_weight_list):
        model.eval()
        result.append(model(input) * torch.tensor(weight).to(input.device))
    result = torch.stack(result, dim=0)
    result = torch.sum(result, dim=0)
    print(result.shape)
    _, predict_label = torch.max(result.data, 1)
    return predict_label.item()


def plot_action_cls(cls, frame):
    
    height, width = frame.shape[:2]
    # 创建与原图相同大小的黑色覆盖层
    overlay = np.zeros_like(frame)

    # 在覆盖层上绘制白色（或黑色）矩形
    top_left = (width - 600, 0)
    bottom_right = (width, 400)
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), thickness=-1)

    # 设置透明度（alpha 控制覆盖层的透明度，beta 控制原图的透明度）
    alpha = 0.5  # 越大越不透明
    beta = 1 - alpha

    # 图像混合
    output = cv2.addWeighted(frame, alpha, overlay, beta, 0)
    # 在矩形内绘制文本
    cv2.putText(output, f'Action: {cls}', (top_left[0] + 20, top_left[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--tracknet_file', type=str, default="/home/awsdjikl/TrackNetV3/exp/TrackNet_best.pt", help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='range of start second and end second of the video for generating median image')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=True, help='whether to process large video')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    
    parser.add_argument(
        '--phase', default='test', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')
    
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    
    args = parser.parse_args()

    args.seed = 42
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_workers = args.batch_size if args.batch_size <= 16 else 16
    video_file = args.video_file
    video_name = video_file.split('/')[-1][:-4]
    video_range = args.video_range if args.video_range else None
    large_video = args.large_video
    args.save_dir = Path(args.video_file).parent
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load model
    tracknet_ckpt = torch.load(args.tracknet_file, weights_only=False)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file, weights_only=False)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    cap = cv2.VideoCapture(args.video_file)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # Test on TrackNet
    tracknet.eval()
    seq_len = tracknet_seq_len
    if args.eval_mode == 'nonoverlap':
        # Create dataset with non-overlap sampling
        if large_video:
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode,
                                            max_sample_num=args.max_sample_num, video_range=video_range)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            print(f'Video length: {dataset.video_len}')
        else:
            # Sample all frames from video
            frame_list = generate_frames(args.video_file)
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:,:,:,::-1], padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
                # print(y_pred.shape)
            # Predict
            tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
    else:
        # Create dataset with overlap sampling for temporal ensemble
        if large_video:
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode,
                                            max_sample_num=args.max_sample_num, video_range=video_range)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            video_len = dataset.video_len
            print(f'Video length: {video_len}')
            
        else:
            # Sample all frames from video
            frame_list = generate_frames(args.video_file)
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:,:,:,::-1])
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            video_len = len(frame_list)
        
        # Init prediction buffer params
        num_sample, sample_count = video_len - seq_len + 1, 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len)  # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len - 1, -1, -1)  # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        weight = get_ensemble_weight(seq_len, args.eval_mode)
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
                # print(y_pred.shape)
            
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    # Imcomplete buffer
                    y_pred = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    # General case
                    y_pred = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)
                
                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # Last batch
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            # Predict
            tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-buffer_size:]

    # assert video_len == len(tracknet_pred_dict['Frame']), 'Prediction length mismatch'
    # Test on TrackNetV3 (TrackNet + InpaintNet)
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h * 0.05)
        inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

        if args.eval_mode == 'nonoverlap':
            # Create dataset with non-overlap sampling
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)  # replace predicted coordinates with inpainted coordinates
                
                # Thresholding
                th_mask = ((coor_inpaint[:,:, 0] < COOR_TH) & (coor_inpaint[:,:, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.
                
                # Predict
                tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
        else:
            # Create dataset with overlap sampling for temporal ensemble
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            weight = get_ensemble_weight(seq_len, args.eval_mode)

            # Init buffer params
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len)  # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len - 1, -1, -1)  # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)
                
                # Thresholding
                th_mask = ((coor_inpaint[:,:, 0] < COOR_TH) & (coor_inpaint[:,:, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        coor_inpaint = coor_inpaint_buffer[batch_i + b, frame_i].sum(0)
                        coor_inpaint /= (sample_count + 1)
                    else:
                        # General case
                        coor_inpaint = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last input sequence
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        
                        for f in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0)
                            coor_inpaint /= (seq_len - f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                # Thresholding
                th_mask = ((ensemble_coor_inpaint[:,:, 0] < COOR_TH) & (ensemble_coor_inpaint[:,:, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

    # Write csv file
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
    write_pred_csv(pred_dict, save_file=out_csv_file)

    # Write video with predicted coordinates
    if args.output_video:
        write_pred_video(video_file, pred_dict, save_file=out_video_file, traj_len=args.traj_len)

    print('Track Ball Done.')
    
    tracknet = None
    torch.cuda.empty_cache()

    # ball_data = pd.read_csv("/home/awsdjikl/TrackNetV3/prediction/game3_ball.csv")
    # video_file = "/home/awsdjikl/TrackNetV3/prediction/game3.mp4"
    # plot_ball_frame(ball_frame)

    ball_data = pd.read_csv(out_csv_file)
    all_strike_positions = get_shot_frame(ball_data)
    # video_list = slice_videos(video_file, all_strike_positions, lenth=30)
    video_list = slice_videos(video_file, all_strike_positions, lenth=WINDOW_SIZE)
    print('Slice Shot Frame Done.')
    # yolo_model = YOLO('/home/awsdjikl/TrackNetV3/models/yolo11n-pose.pt', "pose")
    model_list = []
    config_list = [
        "./TrackNetV3/GCN/config/j.yaml",
        "./TrackNetV3/GCN/config/b.yaml",
        "./TrackNetV3/GCN/config/jm.yaml",
        "./TrackNetV3/GCN/config/bm.yaml",
    ]
    work_dir_list = [
        "./TrackNetV3/GCN/work_dir/j",
        "./TrackNetV3/GCN/work_dir/b",
        "./TrackNetV3/GCN/work_dir/jm",
        "./TrackNetV3/GCN/work_dir/bm",
    ]
    for c, wd in zip(config_list, work_dir_list):
        model_parser = get_parser()
        p = model_parser.parse_args()
        p.config = c
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        model_parser.set_defaults(**default_arg)
        model_args = model_parser.parse_args()
        model_args.work_dir = wd
        model_args.weight = os.path.join(wd, 'runs-65-845.pt')
        p = Processor(model_args)
        p.model.eval()
        model_list.append(p.model)
    model_weight_list = [2.0, 0.8245193386538215, 0.4757473103679432, 0.0001]
    for video in video_list:
        keypoint_list, frame_list = get_vitpose_keypoint(video, ball_data, model_list)
        
        # 保存骨骼数据
        # 将骨骼数据导入st-gcn
        # show_all_openpose_pose(video)
        # get_yolo11_keypoint(video, ball_data, yolo_model)

