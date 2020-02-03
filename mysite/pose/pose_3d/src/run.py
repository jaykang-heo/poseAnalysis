import argparse
import ast
import logging
import time
import os
from .common import MPIIPart, CocoPart
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .estimator import TfPoseEstimator
from .networks import get_graph_path, model_wh
from .lifting.prob_model import Prob3dPose
from .lifting.draw import plot_pose


from statistics import mode

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
path = BASE_DIR.replace('\\'[0], '/')


def upper_arm_angle_cal(pose_3d):
    a = np.array([pose_3d[0][0][7], pose_3d[0][1][7], pose_3d[0][2][7]])
    b = np.array([pose_3d[0][0][8], pose_3d[0][1][8], pose_3d[0][2][8]])
    c = np.array([pose_3d[0][0][11], pose_3d[0][1][11], pose_3d[0][2][11]])
    d = np.array([pose_3d[0][0][12], pose_3d[0][1][12], pose_3d[0][2][12]])

    ba = a - b
    bc = d - c

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    x = np.array([pose_3d[0][0][7], pose_3d[0][1][7], pose_3d[0][2][7]])
    y = np.array([pose_3d[0][0][8], pose_3d[0][1][8], pose_3d[0][2][8]])
    z = np.array([pose_3d[0][0][14], pose_3d[0][1][14], pose_3d[0][2][14]])
    w = np.array([pose_3d[0][0][15], pose_3d[0][1][15], pose_3d[0][2][15]])

    yx = x - y
    yz = w - z

    cosine_angle1 = np.dot(yx, yz) / (np.linalg.norm(yx) * np.linalg.norm(yz))
    angle1 = np.arccos(cosine_angle1)

    return np.degrees(angle), np.degrees(angle1)


def neck_angle_cal(pose_3d):
    x = np.array([pose_3d[0][0][7], pose_3d[0][1][7], pose_3d[0][2][7]])
    y = np.array([pose_3d[0][0][8], pose_3d[0][1][8], pose_3d[0][2][8]])
    z = np.array([pose_3d[0][0][9], pose_3d[0][1][9], pose_3d[0][2][9]])

    yx = x - y
    yz = z - y

    cosine_angle = np.dot(yx, yz) / (np.linalg.norm(yx) * np.linalg.norm(yz))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def lower_arm_angle_cal(pose_3d):

    a = np.array([pose_3d[0][0][11], pose_3d[0][1][11], pose_3d[0][2][11]])
    b = np.array([pose_3d[0][0][12], pose_3d[0][1][12], pose_3d[0][2][12]])
    c = np.array([pose_3d[0][0][13], pose_3d[0][1][13], pose_3d[0][2][13]])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    left_arm = np.arccos(cosine_angle)

    x = np.array([pose_3d[0][0][14], pose_3d[0][1][14], pose_3d[0][2][14]])
    y = np.array([pose_3d[0][0][15], pose_3d[0][1][15], pose_3d[0][2][15]])
    z = np.array([pose_3d[0][0][16], pose_3d[0][1][16], pose_3d[0][2][16]])

    yx = x - y
    yz = z - y

    cosine_angle = np.dot(yx, yz) / (np.linalg.norm(yx) * np.linalg.norm(yz))
    right_arm = np.arccos(cosine_angle)
    return np.degrees(left_arm), np.degrees(right_arm)


def interpolation(pose_interp):

    result = []
    pose_interp[0] = interp1d(np.arange(len(pose_interp[0])), pose_interp[0])
    pose_interp[1] = interp1d(np.arange(len(pose_interp[1])), pose_interp[1])
    pose_interp[2] = interp1d(np.arange(len(pose_interp[2])), pose_interp[2])
    pose_interp[3] = interp1d(np.arange(len(pose_interp[3])), pose_interp[3])
    pose_interp[4] = interp1d(np.arange(len(pose_interp[4])), pose_interp[4])

    # print(pose_interp[0].y)

    df = pd.DataFrame(pose_interp[0].y)
    df = df.interpolate(method='spline', order=1, limit=1000, limit_direction='both')

    df1 = pd.DataFrame(pose_interp[1].y)
    df1 = df1.interpolate(method='spline', order=1, limit=1000, limit_direction='both')

    df2 = pd.DataFrame(pose_interp[2].y)
    df2 = df2.interpolate(method='spline', order=1, limit=1000, limit_direction='both')

    df3 = pd.DataFrame(pose_interp[3].y)
    df3 = df3.interpolate(method='spline', order=1, limit=1000, limit_direction='both')

    df4 = pd.DataFrame(pose_interp[4].y)
    df4 = df4.interpolate(method='spline', order=1, limit=1000, limit_direction='both')

    df = df.values.tolist()
    df1 = df1.values.tolist()
    df2 = df2.values.tolist()
    df3 = df3.values.tolist()
    df4 = df4.values.tolist()
    result.append([df, df1, df2, df3, df4])
    return result, pose_interp


def calculate_point(result):
    points = []
    # for angles in result:
    arm1 = []
    arm2 = []
    for check1, check2 in zip(result[0], result[1]):
        if check1[0] < 30:
            arm1_number = 1
            arm1.append(arm1_number)
        elif 30 <= check1[0]:
            arm1_number = 2
            arm1.append(arm1_number)
        else:
            pass
            # arm1_number = 1
            # arm1.append(arm1_number)
        if check2[0] < 30:
            arm2_number = 1
            arm2.append(arm2_number)
        elif 30 <= check2[0]:
            arm2_number = 2
            arm2.append(arm2_number)
        else:
            pass
            # arm2_number = 1
            # arm2.append(arm2_number)

    neck = []
    for neck_check in result[2]:
        if 160 > neck_check[0] > 95:
            neck_number = 1
            neck.append(neck_number)
        elif 95 >= neck_check[0]:
            neck_number = 2
            neck.append(neck_number)
        elif 160 <= neck_check[0]:
            neck_number = 3
            neck.append(neck_number)
        else:
            pass
            # neck_number = 1
            # neck.append(neck_number)

    left_upper_arm = []
    right_upper_arm = []
    for left_upper_arm_check, right_upper_arm_check in zip(result[3], result[4]):
        if 60 > left_upper_arm_check[0]:
            left_upper_arm_number = 1
            left_upper_arm.append(left_upper_arm_number)
        elif 90 > left_upper_arm_check[0] >= 60:
            left_upper_arm_number = 2
            left_upper_arm.append(left_upper_arm_number)
        elif left_upper_arm_check[0] > 90:
            left_upper_arm_number = 3
            left_upper_arm.append(left_upper_arm_number)
        else:
            pass
            # left_upper_arm_number = 1
            # left_upper_arm.append(left_upper_arm_number)
        if 60 > right_upper_arm_check[0]:
            right_upper_arm_number = 1
            right_upper_arm.append(right_upper_arm_number)
        elif 90 > right_upper_arm_check[0] >= 60:
            right_upper_arm_number = 2
            right_upper_arm.append(right_upper_arm_number)
        elif right_upper_arm_check[0] >= 90:
            right_upper_arm_number = 3
            right_upper_arm.append(right_upper_arm_number)
        else:
            pass
            # right_upper_arm_number = 1
            # right_upper_arm.append(right_upper_arm_number)
    points.append([arm1, arm2, neck, left_upper_arm, right_upper_arm])
    return points


def generate_video(points, frame_number, title):
    # print(points.y)
    result = []
    # for i in points.y:
    #     result.append(i[0])
    # result = np.array(result)
    # print(result)
    # frame_list = list(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    # frame_list = np.array(frame_list)
    # arm1 = points[0]
    # arm1 = np.array(arm1)
    # arm2 = points[1]
    # arm2 = np.array(arm2)
    # neck = points[2]
    # neck = np.array(neck)
    # left_upper = points[3]
    # left_upper = np.array(left_upper)
    # right_upper = points[4]
    # right_upper = np.array(right_upper)

    fig = plt.figure()

    # cap1 = cv2.VideoCapture(path + '/media/export_left_13088_13989_output.mp4')
    # input_fps = cap1.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(path + '/media/combined.mp4', fourcc,
    #                       input_fps,
    #                       (int(cap1.get(3)), int(cap1.get(4))))

    # print(neck)
    # print(type(points.y))
    # print(points.y.shape)
    result = np.array(points.y)
    # print(result.shape)

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_xticks([])
    # ax.spines['bottom'].set_visible(False)
    ax.set_title(title, fontsize=6)
    # print(len(points.y))
    ax.plot(np.array(frame_number), result, linestyle='-', marker='o', markersize=1)
    # ax = fig.add_subplot(5, 1, 2)
    # ax.set_xticks([])
    # ax.spines['bottom'].set_visible(False)
    # ax.set_title('right arm', fontsize=6)
    # ax.plot(len(left_upper), left_upper, 'k')
    # ax = fig.add_subplot(5, 1, 3)
    # ax.set_xticks([])
    # ax.spines['bottom'].set_visible(False)
    # ax.set_title('neck', fontsize=6)
    # ax.plot(len(right_upper), right_upper, 'k')
    # ax = fig.add_subplot(5, 1, 4)
    # ax.set_xticks([])
    # ax.spines['bottom'].set_visible(False)
    # ax.set_title('upper left arm', fontsize=6)
    # ax.plot(frame_list[0:cur], left_upper[0:cur], 'k')
    # ax = fig.add_subplot(5, 1, 5)
    # ax.set_xticks([])
    # ax.spines['bottom'].set_visible(False)
    # ax.set_title('upper right arm', fontsize=6)
    # ax.plot(frame_list[0:cur], right_upper[0:cur], 'k')
    # cap1.set(1, cur)
    # ret, frame = cap1.read()

    # redraw the canvas
    fig.canvas.draw()
    plt.savefig(path+'/'+title, dpi=250)

    # convert canvas to image
    # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #
    # # img is rgb, convert to opencv's default bgr
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = cv2.resize(img, (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    #
    # # combine with video image frame
    # vis = np.concatenate((frame, img), axis=1)
    # vis = cv2.resize(vis, (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    # out.write(vis)
    # out.release()
    # cap1.release()


def get_numbers(humans, image, box):
    image_w, image_h = image.shape[:2]
    human_nums = [0]
    if box is not '\n':
        # different number for every loop
        numbers_included = 0
        coords = box.split(',')
        num = int(coords[-1].split('\n')[0].split(';')[0])
        coords = coords[:-1]
        x = int(coords[0])
        y = int(coords[1])
        width = int(coords[2])
        height = int(coords[3])
        for human in humans:
            compare_included = 0
            for j in range(CocoPart.Background.value):
                if j not in human.body_parts.keys():
                    continue
                body_part = human.body_parts[j]
                pose_x = int(body_part.x * image_w + 0.5)
                pose_y = int(body_part.y * image_h + 0.5)
                if x < pose_x < width and y < pose_y < height:
                    compare_included += 1
            if compare_included > numbers_included:
                numbers_included = compare_included
                if compare_included > 5:
                    human_nums.append(human)
        if human_nums[-1] != 0:
            TfPoseEstimator.draw_humans(image, [human_nums[-1]], imgcopy=False)
            cv2.rectangle(image, (int(x), int(y)), (int(width), int(height)), [0, 0, 255], 4)
            cv2.putText(image, str(num), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 2, [0, 0, 255])
    return [human_nums[-1]]


def pose(video_file):
    model = path +'/mysite/pose/pose_3d/models/graph_opt.pb'
    w, h = 432, 368

    #######################################################################
    #######################################################################
    #######################################################################
    video_title = video_file[0]
    video_title = video_title.split('.mp4')[0]
    #######################################################################
    #######################################################################
    #######################################################################

    e = TfPoseEstimator(model, target_size=(w, h))
    video = path + '/'+video_title+'.mp4'
    cap = cv2.VideoCapture(video)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    interps = []
    # left arm, right_arm, left upper arm, right upper arm, neck
    pose_interp = [[], [], [], [], []]
    yolo_text_file = path + '/'+video_title+'.txt'
    yolo_text_file1 = path + '/'+video_title+'_new.txt'
    d = open(yolo_text_file, "r")
    new = open(yolo_text_file1, 'w')
    for line in d.readlines():
        if line == '\n':
            new.write('\n')
        else:
            line = line.split(';')
            for dine in line[:-1]:
                han = dine.split(',')
                num = han[-1]
                x = han[0]
                y = han[1]
                width = han[2]
                height = han[3]
                for pick in video_file[1]:
                    if str(num) == pick:
                        new.write(str(x)+','+str(y)+','+str(width)+','+str(height)+','+str(num)+'\n')
    d.close()
    new.close()

    f = open(yolo_text_file1, 'r')
    count = 0
    missing_line = 0
    pose_not = 0
    truth_state = []
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = path + '/'+video_title+'_pose.mp4'
    out = cv2.VideoWriter(path + '/'+video_title+'_pose.mp4', fourcc,
                          input_fps,
                          (int(cap.get(3)), int(cap.get(4))))
    frame_number = []
    for box in f.readlines():

        ret, frame = cap.read()
        cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        logging.info(
            "Pose estimation " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) + ' / ' + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # estimate human poses from a single image !
        # frame = common.read_imgfile(frame, None, None)
        humans = e.inference(frame, scales=[None])
        # frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)

        # get a list of tracking numbers according to human coordinates
        human_nums = get_numbers(humans, frame, box)
        out.write(frame)
        truth_state.append(len(humans))
        if box is not '\n':
            coords = box.split(',')
            read_or_not = int(coords[-1])
        else:
            read_or_not = 0
            missing_line += 1
        if str(read_or_not) in video_file[1]:
            frame_number.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if human_nums[-1] is 0:
                print("In line, but human is not detected ")
                pose_interp[0].append(None)
                pose_interp[1].append(None)
                pose_interp[2].append(None)
                pose_interp[3].append(None)
                pose_interp[4].append(None)
                pose_not += 1
            else:
                count += 1
                # 3d modeling
                model_3d_path = path+'/mysite/pose/pose_3d/src/lifting/models/prob_model_params.mat'
                poseLifting = Prob3dPose(model_3d_path)
                image_h, image_w = frame.shape[:2]
                standard_w = 640
                standard_h = 480

                pose_2d_mpiis = []
                visibilities = []

                for human in humans:
                    pose_2d_mpii, visibility = MPIIPart.from_coco(human)
                    pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
                    visibilities.append(visibility)

                pose_2d_mpiis = np.array(pose_2d_mpiis)
                visibilities = np.array(visibilities)
                transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
                if transformed_pose2d is None:
                    pose_interp[0].append(None)
                    pose_interp[1].append(None)
                    pose_interp[2].append(None)
                    pose_interp[3].append(None)
                    pose_interp[4].append(None)
                else:
                    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
                    # if pose_3d is not None:
                    # print("Boolean for equal if pose_3d and humans: ", len(pose_3d) == len(humans))
                    if pose_3d is None:
                        for pose in pose_interp:
                            pose.append(None)
                    else:
                        left_arm, right_arm = lower_arm_angle_cal(pose_3d)
                        neck_angle = neck_angle_cal(pose_3d)
                        left_upper_arm_angle, right_upper_arm_angle = upper_arm_angle_cal(pose_3d)
                        pose_interp[0].append(int(left_arm))
                        pose_interp[1].append(int(right_arm))
                        pose_interp[2].append(int(neck_angle))
                        pose_interp[3].append(int(left_upper_arm_angle))
                        pose_interp[4].append(int(right_upper_arm_angle))
    out.release()

    # do interpolation
    result, pose_interp = interpolation(pose_interp)

    # make video in media folder
    pose_neck = open(path+'/'+video_title+'_neck_pose.txt', 'w')
    pose_left_upper_arm = open(path+'/'+video_title+'_left_upper_arm.txt', 'w')
    pose_right_upper_arm = open(path+'/'+video_title+'_right_upper_arm.txt', 'w')
    for pp_neck, pp_left, pp_right in zip(pose_interp[2].y, pose_interp[3].y, pose_interp[4].y):
        pose_neck.write(str(pp_neck)+'\n')
        pose_left_upper_arm.write(str(pp_left)+'\n')
        pose_right_upper_arm.write(str(pp_right)+'\n')
    pose_neck.close()
    pose_left_upper_arm.close()
    pose_right_upper_arm.close()

    neck_path = path + '/' + video_title+'_neck_result_with_nan.png'
    left_arm_path = path + '/' + video_title+'_left_arm_result_with_nan.png'
    right_arm_path = path + '/' + video_title+'_right_arm_result_with_nan.png'

    generate_video(pose_interp[2], frame_number, video_title+'_neck_result_with_nan.png')
    generate_video(pose_interp[3], frame_number, video_title+'_left_arm_result_with_nan.png')
    generate_video(pose_interp[4], frame_number, video_title+'_right_arm_result_with_nan.png')
    # generate_video(pose_interp[0], 'interpolated_result_with_nan.png')

    # calculate points according to different angles
    # points = calculate_point(result[0])
    #
    # # make video in media folder
    # # generate_video(points[0], 'interpolated_result_without_nan.png')
    #
    # neck_neutral = points[0][2].count(1)
    # neck_flex = points[0][2].count(2)
    # neck_ext = points[0][2].count(3)
    # neck_miss = cap.get(cv2.CAP_PROP_FRAME_COUNT)-count
    # neck_total = neck_neutral+neck_flex+neck_ext+neck_miss
    #
    # upper_arm_neutral = points[0][3].count(1)
    # upper_arm_flex = points[0][3].count(2)
    # upper_arm_ext = points[0][3].count(3)
    # upper_arm_miss = cap.get(cv2.CAP_PROP_FRAME_COUNT)-count
    # upper_arm_total = upper_arm_neutral+upper_arm_ext+upper_arm_flex+upper_arm_miss
    #
    # upper_arm1_neutral = points[0][4].count(1)
    # upper_arm1_flex = points[0][4].count(2)
    # upper_arm1_ext = points[0][4].count(3)
    # upper_arm1_miss = cap.get(cv2.CAP_PROP_FRAME_COUNT)-count
    # upper_arm1_total = upper_arm1_ext+upper_arm1_neutral+upper_arm1_ext+upper_arm1_miss
    #
    # miss = count / cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # pose_total = ''
    # pose_result_text_path = path+'/'+video_file[0]+'_pose_result.txt'
    # pose_result = open(pose_result_text_path, 'w')
    # total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #
    # pose_total += 'Total Frame: ' + str(total_frame)+'\n'
    # pose_total += 'Box Not Detected: ' + str(missing_line) + '\n'
    # pose_total += 'Pose Not Detected (Box Detected): ' + str(pose_not) + '\n'
    # neck = str(neck_neutral*100/neck_total)+' / ' + str(neck_flex*100/neck_total)+' / ' + str(neck_ext*100/neck_total)+' / ' + str(neck_miss*100/neck_total)
    # upper_arm = str(upper_arm_neutral*100/neck_total)+' / ' + str(upper_arm_flex*100/neck_total)+' / ' + str(upper_arm_ext*100/neck_total)+' / ' + str(upper_arm_miss*100/neck_total)
    # upper_arm1 = str(upper_arm1_neutral*100/neck_total)+' / ' + str(upper_arm1_flex*100/neck_total)+' / ' + str(upper_arm1_ext*100/neck_total)+' / ' + str(upper_arm1_miss*100/neck_total)
    # pose_total += 'Neck: ' + neck+'\n'+'Upper Arm(Shoulder): '+upper_arm+'\n'+'Upper Arm1(Shoulder): '+upper_arm1+'\n'
    # pose_result.write(pose_total)
    # pose_result.close()
    return [0, neck_path, left_arm_path, right_arm_path, video_path, pose_neck, pose_left_upper_arm, pose_right_upper_arm]



