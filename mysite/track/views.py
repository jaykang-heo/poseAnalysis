from django.shortcuts import render
import logging
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = BASE_DIR.replace('\\'[0], '/')
box_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def track(video_file):
    import cv2
    import numpy as np
    from .yolo import YOLO
    from .deep_sort import preprocessing
    from .deep_sort import nn_matching
    from .deep_sort.detection import Detection
    from .deep_sort.tracker import Tracker
    from .tools import generate_detections as gdet
    from keras import backend as K
    from PIL import Image

    K.clear_session()
    yolo_track = YOLO()

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    path = BASE_DIR.replace('\\'[0], '/')
    model_filename = path + '/mysite/track/model_data/mars-small128.pb'
    # uploaded_file = request.FILES['document']
    video_title = video_file
    video = path + '/'+video_title
    video_title1 = video_title.split('.mp4')[0]
    text_name = video_title1+'.txt'

    text_path = path + '/' + video_title1+'_output.txt'
    box_file = open(text_path, "w")
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    total_text = ''
    tracker = Tracker(metric)
    video_capture = cv2.VideoCapture(video)
    input_fps = video_capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    write_video_path = path + '/'+video_title1+'_output.mp4'
    write_name = video_title1 + '_output.mp4'
    out = cv2.VideoWriter(write_video_path, fourcc,
                          input_fps,
                          (int(video_capture.get(3)), int(video_capture.get(4))))
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolo_track.detect_image(image)
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        total = ''
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            x = str(int(bbox[0]))
            y = str(int(bbox[1]))
            width = str(int(bbox[2]))
            height = str(int(bbox[3]))
            id = str(track.track_id)

            total += x + ',' + y + ',' + width + ',' + height + ',' + id + ';'

            cv2.rectangle(frame, (int(x), int(y)), (int(width), int(height)), box_colors[int(0)], 2)
            cv2.putText(frame, str(id), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 2, box_colors[int(0)])

        logging.info("People Detection " + str(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) + ' / ' + str(
            video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        out.write(frame)
        total_text += str(total) + '\n'

    box_file.write(total_text)
    box_file.close()
    video_capture.release()
    out.release()

