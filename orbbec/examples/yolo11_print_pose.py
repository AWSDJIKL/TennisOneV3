from ultralytics import YOLO
from ultralytics.engine.results import Keypoints
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from tqdm.contrib import tzip
import argparse
from tqdm import trange
from PIL import Image
import glob
import copy
import sys

sys.path.append(os.getcwd())
from common.utils import *
from common.camera import *
from model.mixste.hot_mixste import Model
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose

# from panel.tests.manual.models import video
# from dask.dataframe.tests.test_categorical import frames
print(os.getcwd())

point_id_dict = {
    0: "鼻子",
    1: "左眼",
    2: "右眼",
    3: "左耳",
    4: "右耳",
    5: "左肩",
    6: "右肩",
    7: "左肘",
    8: "右肘",
    9: "左腕",
    10: "右腕",
    11: "左髋关节",
    12: "右髋关节",
    13: "左膝",
    14: "右膝",
    15: "左脚踝",
    16: "右脚踝", }


def calculate_angle(a, b, c):
    # 输入：三个点的坐标 (x, y)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1, 1))  # 避免数值溢出
    return np.degrees(angle)


def calculate_3d_angle(a, b, c):
    # 输入：三个点的三维坐标 (x, y, z)
    ba = a - b  # 向量 BA
    bc = c - b  # 向量 BC
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * (np.linalg.norm(bc) + 1e-6))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))
    return angle


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


# def show3Dpose(vals, ax, fix_z):
#     ax.view_init(elev=15., azim=70)

#     colors = [(138 / 255, 201 / 255, 38 / 255),
#             (255 / 255, 202 / 255, 58 / 255),
#             (25 / 255, 130 / 255, 196 / 255)]

#     I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
#     J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
#     # 颜色编号
#     LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

#     for i in np.arange(len(I)):
#         x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
#         ax.plot(x, y, z, lw=3, color=colors[LR[i] - 1])

#     RADIUS = 0.72

#     xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
#     ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
#     ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

#     if fix_z:
#         left_z = max(0.0, -RADIUS + zroot)
#         right_z = RADIUS + zroot
#         # ax.set_zlim3d([left_z, right_z])
#         ax.set_zlim3d([0, 1.5])
#     else:
#         ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])

#     ax.set_aspect('equal')  # works fine in matplotlib==2.2.2 or 3.7.1

#     white = (1.0, 1.0, 1.0, 0.0)
#     ax.xaxis.set_pane_color(white)
#     ax.yaxis.set_pane_color(white)
#     ax.zaxis.set_pane_color(white)

#     ax.tick_params('x', labelbottom=False)
#     ax.tick_params('y', labelleft=False)
#     ax.tick_params('z', labelleft=False)

#     # 定义需要计算角度的关节三元组
#     angle_joints = [
#         (5, 7, 9, "L_Elbow"),  # 左肘
#         (6, 8, 10, "R_Elbow"),  # 右肘
#         (11, 13, 15, "L_Knee"),  # 左膝
#         (12, 14, 16, "R_Knee")  # 右膝
#     ]

#     # # 定义需要标注的关节三元组（COCO格式）
#     # angle_joints = [
#     #     (11, 12, 13, "L_Elbow"),  # 左肘（肩-肘-腕）
#     #     (14, 15, 16, "R_Elbow"),  # 右肘
#     #     (1, 2, 3, "L_Knee"),  # 左膝（髋-膝-踝）
#     #     (4, 5, 6, "R_Knee")  # 右膝
#     # ]

#     for joint in angle_joints:
#         a_idx, b_idx, c_idx, label = joint
#         a = vals[a_idx]
#         b = vals[b_idx]
#         c = vals[c_idx]

#         # # 检查置信度（如果可用）
#         # if scores is not None:
#         #     if scores[a_idx] < conf_threshold or scores[b_idx] < conf_threshold or scores[c_idx] < conf_threshold:
#         #         continue

#         # 计算3D角度
#         angle = calculate_3d_angle(a, b, c)

#         # 在3D空间中标注角度（以b点为中心）
#         text_pos = (b[0], b[1], b[2] + 0.1)  # 在关节点上方偏移
#         ax.text(
#             *text_pos,
#             f"{label}:{angle:.1f}",
#             color='red',
#             fontsize=8,
#             ha='center',
#             va='bottom'
#         )

# def get_pose2D(video_path, output_dir):
#     cap = cv2.VideoCapture(video_path)
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

#     print('\nGenerating 2D pose...')
#     with torch.no_grad():
#         # the first frame of the video should be detected a person
#         keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
#     keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
#     re_kpts = revise_kpts(keypoints, scores, valid_frames)
#     print('Generating 2D pose successfully!')

#     output_dir += 'input_2D/'
#     os.makedirs(output_dir, exist_ok=True)

#     output_npz = output_dir + 'input_keypoints_2d.npz'
#     np.savez_compressed(output_npz, reconstruction=keypoints)

# def adjust_size(img, target_h):
#     w, h = img.size
#     new_w = int(w * (target_h / h))
#     return img.resize((new_w, target_h))

# def img2video(video_path, output_dir):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#     # names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))

#     # 将pose2D和pose3D的图片合并成一个视频
#     pose2d_img_list = sorted(glob.glob(os.path.join(output_dir + 'pose2D/', '*.png')))
#     pose3d_img_list = sorted(glob.glob(os.path.join(output_dir + 'pose3D/', '*.png')))
#     print('pose2d_img_list: ', pose2d_img_list)
#     print('pose3d_img_list: ', pose3d_img_list)
#     img_2d = Image.open(pose2d_img_list[0])
#     img_3d = Image.open(pose3d_img_list[0])
#     video_size = (0, 0)
#     if img_2d.size[1] < img_3d.size[1]:
#         img_2d = adjust_size(img_2d, img_3d.size[1])
#         video_size = (img_2d.size[0] + img_3d.size[0], img_2d.size[1])
#     else:
#         img_3d = adjust_size(img_3d, img_2d.size[1])
#         video_size = (img_2d.size[0] + img_3d.size[0], img_3d.size[1])
#     # img = cv2.imread(names[0])
#     # size = (img.shape[1], img.shape[0])

#     videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, video_size)

#     for img_2d_path, img_3d_path in tzip(pose2d_img_list, pose3d_img_list):
#         img_2d = Image.open(img_2d_path)
#         img_3d = Image.open(img_3d_path)
#         if img_2d.size[1] < img_3d.size[1]:
#             img_2d = adjust_size(img_2d, img_3d.size[1])
#         else:
#             img_3d = adjust_size(img_3d, img_2d.size[1])
#         # 水平拼接
#         combined = Image.new('RGB', (img_2d.size[0] + img_3d.size[0], img_2d.size[1]))
#         combined.paste(img_2d, (0, 0))
#         combined.paste(img_3d, (img_2d.size[0], 0))

#         # 转换为OpenCV格式（BGR）
#         frame = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

#         # 写入视频
#         videoWrite.write(frame)

#     videoWrite.release()

# def showimage(ax, img):
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.axis('off')
#     ax.imshow(img)

# def get_pose3D(video_path, output_dir, fix_z):
#     args, _ = argparse.ArgumentParser().parse_known_args()
#     args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, 243
#     args.token_num, args.layer_index = 81, 3
#     args.pad = (args.frames - 1) // 2
#     args.previous_dir = 'checkpoint/pretrained/hot_mixste'
#     args.n_joints, args.out_joints = 17, 17

#     # # Reload
#     model = Model(args).cuda()

#     model_dict = model.state_dict()
#     # Put the pretrained model in 'checkpoint/pretrained/hot_mixste'
#     model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

#     pre_dict = torch.load(model_path)
#     model_dict = model.state_dict()
#     state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
#     model_dict.update(state_dict)
#     model.load_state_dict(model_dict)

#     model.eval()

#     # # input
#     keypoints = np.load(output_dir + 'input_2D/input_keypoints_2d.npz', allow_pickle=True)['reconstruction']

#     cap = cv2.VideoCapture(video_path)
#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     n_chunks = video_length // args.frames + 1
#     offset = (n_chunks * args.frames - video_length) // 2

#     ret, img = cap.read()
#     img_size = img.shape

#     # # 3D
#     print('\nGenerating 3D pose...')
#     frame_sum = 0
#     for i in tqdm(range(n_chunks)):
#         # # input frames
#         start_index = i * args.frames - offset
#         end_index = (i + 1) * args.frames - offset

#         low_index = max(start_index, 0)
#         high_index = min(end_index, video_length)
#         pad_left = low_index - start_index
#         pad_right = end_index - high_index

#         if pad_left != 0 or pad_right != 0:
#             input_2D_no = np.pad(keypoints[0][low_index:high_index], ((pad_left, pad_right), (0, 0), (0, 0)), 'edge')
#         else:
#             input_2D_no = keypoints[0][low_index:high_index]

#         # joints_left = [4, 5, 6, 11, 12, 13]
#         # joints_right = [1, 2, 3, 14, 15, 16]

#         joints_left = [5, 7, 9, 11, 13, 15]
#         joints_right = [6, 8, 10, 12, 14, 16]

#         input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])

#         input_2D_aug = copy.deepcopy(input_2D)
#         input_2D_aug[:,:, 0] *= -1
#         input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
#         input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

#         input_2D = input_2D[np.newaxis,:,:,:,:]

#         input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

#         N = input_2D.size(0)

#         # # estimation
#         with torch.no_grad():
#             output_3D_non_flip = model(input_2D[:, 0])
#             output_3D_flip = model(input_2D[:, 1])

#         output_3D_flip[:,:,:, 0] *= -1
#         output_3D_flip[:,:, joints_left + joints_right,:] = output_3D_flip[:,:, joints_right + joints_left,:]

#         output_3D = (output_3D_non_flip + output_3D_flip) / 2

#         if pad_left != 0 and pad_right != 0:
#             output_3D = output_3D[:, pad_left:-pad_right]
#             input_2D_no = input_2D_no[pad_left:-pad_right]
#         elif pad_left != 0:
#             output_3D = output_3D[:, pad_left:]
#             input_2D_no = input_2D_no[pad_left:]
#         elif pad_right != 0:
#             output_3D = output_3D[:,:-pad_right]
#             input_2D_no = input_2D_no[:-pad_right]

#         output_3D[:,:, 0,:] = 0
#         post_out = output_3D[0].cpu().detach().numpy()

#         if i == 0:
#             output_3d_all = post_out
#         else:
#             output_3d_all = np.concatenate([output_3d_all, post_out], axis=0)

#         # # h36m_cameras_extrinsic_params in common/camera.py
#         # https://github.com/facebookresearch/VideoPose3D/blob/main/common/custom_dataset.py#L23
#         rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
#         rot = np.array(rot, dtype='float32')
#         post_out = camera_to_world(post_out, R=rot, t=0)

#         # # 2D
#         for j in range(low_index, high_index):
#             jj = j - frame_sum
#             if i == 0 and j == 0:
#                 pass
#             else:
#                 ret, img = cap.read()
#                 img_size = img.shape

#             image = show2Dpose(input_2D_no[jj], copy.deepcopy(img))
#             # image=img

#             output_dir_2D = output_dir + 'pose2D/'
#             os.makedirs(output_dir_2D, exist_ok=True)
#             cv2.imwrite(output_dir_2D + str(('%04d' % j)) + '_2D.png', image)

#             # # 3D
#             fig = plt.figure(figsize=(9.6, 5.4))
#             gs = gridspec.GridSpec(1, 1)
#             gs.update(wspace=-0.00, hspace=0.05)
#             ax = plt.subplot(gs[0], projection='3d')

#             post_out[jj,:, 2] -= np.min(post_out[jj,:, 2])
#             show3Dpose(post_out[jj], ax, fix_z)

#             output_dir_3D = output_dir + 'pose3D/'
#             os.makedirs(output_dir_3D, exist_ok=True)
#             plt.savefig(output_dir_3D + str(('%04d' % j)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')

#         frame_sum = high_index

#     # # save 3D keypoints
#     os.makedirs(output_dir + 'output_3D/', exist_ok=True)
#     output_npz = output_dir + 'output_3D/' + 'output_keypoints_3d.npz'
#     np.savez_compressed(output_npz, reconstruction=output_3d_all)

#     print('Generating 3D pose successfully!')

#     # # # all
#     # image_dir = 'results/'
#     # image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
#     # image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

#     # print('\nGenerating demo...')
#     # for i in tqdm(range(len(image_2d_dir))):
#     #     image_2d = plt.imread(image_2d_dir[i])
#     #     image_3d = plt.imread(image_3d_dir[i])

#     #     # # crop
#     #     edge = (image_2d.shape[1] - image_2d.shape[0]) // 2 - 1
#     #     # image_2d = image_2d[:, edge:image_2d.shape[1] - edge]
#     #     edge_1 = 10
#     #     image_2d = image_2d[edge_1:image_2d.shape[0] - edge_1, edge + edge_1:image_2d.shape[1] - edge - edge_1]

#     #     edge = 130
#     #     image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

#     #     # # show
#     #     font_size = 12
#     #     fig = plt.figure(figsize=(9.6, 5.4))
#     #     ax = plt.subplot(121)
#     #     showimage(ax, image_2d)
#     #     ax.set_title("Input", fontsize=font_size)

#     #     ax = plt.subplot(122)
#     #     showimage(ax, image_3d)
#     #     ax.set_title("Reconstruction", fontsize=font_size)

#     #     # # save
#     #     output_dir_pose = output_dir + 'pose/'
#     #     os.makedirs(output_dir_pose, exist_ok=True)
#     #     plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose.png', dpi=200, bbox_inches='tight')
#     #     plt.close()


def get_yolo11_keypoints(video_path):
    model = YOLO("yolo11x-pose.pt", "pose")  # load a pretrained model (recommended for training)
    # 使用opencv逐帧读取视频
    cap = cv2.VideoCapture(video_path)
    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_list = []
    keypoint_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imshow("frame", frame)
        # cv2.waitKey(1)
        # 预测
        result = model.predict(source=frame)[0]  # predict on an image
        keypoints = result.keypoints.xy[0].cpu().numpy()

        # 如果没有keypoints，说明没有检测到人，放弃这一帧
        if len(keypoints) == 0:
            # print("no keypoints detected, skip this frame")
            continue

        # 遍历keypoints，如果有关键点坐标为0，说明有一些点检测不到，放弃这一帧
        k_id = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        # print("keypoints: ", keypoints)
        # print("keypoints shape: ", keypoints.shape)
        flag = False
        for i in k_id:
            if not keypoints[i][0] > 0 and not keypoints[i][1] > 0:
                print("keypoint {} not detected".format(i))
                flag = True
                break
        if flag:
            # print("have some keypoint not detected, skip this frame")
            continue

        # print(keypoints)
        # print(keypoints.shape)
        keypoint_list.append(keypoints)
        frame_list.append(frame)

    cap.release()
    return frame_list, keypoint_list, width, height


if __name__ == "__main__":
    video_path = "video/output/crop_test.mp4"
    frames, keypoints, w, h = get_yolo11_keypoints(video_path)
    output_video = cv2.VideoWriter("video/output/yolo11_test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for f, k in tzip(frames, keypoints):
        f = show2Dpose(k, f)
        output_video.write(f)
    output_video.release()
