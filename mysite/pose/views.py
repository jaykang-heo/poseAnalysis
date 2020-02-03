from django.shortcuts import render
from .lifting.prob_model import Prob3dPose
from .lifting.draw import plot_pose
from .estimator import TfPoseEstimator
from .networks import get_graph_path, model_wh
import os, cv2
import numpy as np
from . import common

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pose_estimation():
    model = 'graph_opt.pb'
    camera = 0


    w, h = model_wh(model)
    e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    cam = cv2.VideoCapture(camera)
    ret_val, image = cam.read()
    poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')


    while True:
        ret_val, image = cam.read()

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size='432x368')

        # 3d pose estimation
        image_h, image_w = image.shape[:2]
        standard_w = 640
        standard_h = 480

        pose_2d_mpiis = []
        visibilities = []
        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
            visibilities.append(visibility)

        pose_2d_mpiis = np.array(pose_2d_mpiis)
        visibilities = np.array(visibilities)
        transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
        pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)


        # draw 2d pose estimation
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.imshow('tf-pose-estimation result', image)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    return
