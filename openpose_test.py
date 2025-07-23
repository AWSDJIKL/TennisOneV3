import os
import sys
import cv2
from sys import platform
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
print(os.environ['PATH'])
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '\\openpose\\build\\python\\openpose\\Release;'
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '\\openpose\\build\\x64\\Release;'
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '\\openpose\\build\\bin;'
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + ';'
print(os.environ['PATH'])
import pyopenpose as op

print(op)
print("成功引入pyopenpose")

parser = argparse.ArgumentParser()
# 测试图片的路径要改一下，自己修改
parser.add_argument("--image_path",
                    default="./video/tai.jpg",
                    help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()

params["model_folder"] = "./openpose/models"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-', '')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-', '')
        if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# 修改参数
# 修改分辨率，可以降低对显存的占用 （16的倍数）
params["net_resolution"] = "368x256"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
datum = op.Datum()
imageToProcess = cv2.imread(args[0].image_path)
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# Display Image
print("Body keypoints: \n" + str(datum.poseKeypoints))
cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
cv2.waitKey(0)
