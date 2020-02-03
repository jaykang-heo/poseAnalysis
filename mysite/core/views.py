import os, cv2
from django.shortcuts import render
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from ..track.views import track
from ..pose.pose_3d.src.run import pose
from .visualize import merge_shoulder_files, calculate_point, make_excel
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = BASE_DIR.replace('\\'[0], '/')


def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)

        #  path to the media folder
        path_video = context['url']
        # print(path_video)
        # start tacking folder
        track(path_video)
        path_video1 = path_video.split('.mp4')[0]
        path_to_video = path_video1+'_output.mp4'
        path_text = path_video1+'_output.txt'
        context['box_text'] = path_text
        context['box_write'] = path_to_video
        # start pose estimation
        # read_or_not = request.FILES['document']
        # pose_result = pose([path_video, read_or_not])

        # save the media to the dictionary and send it back to the server
        # cap = cv2.VideoCapture(path_video)
        # cap.set(1, 3)  # Where frame_no is the frame you want
        # ret, frame = cap.read()
        # print(path)
        # cv2.imwrite(os.path.join(path+'/media', 'sample.jpg'), frame)
        # context['sample_img'] = fs.url('sample.jpg')

    return render(request, 'home.html', context)


def pose_upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        uploaded_text = request.POST.get('textfield', '')
        uploaded_file = str(uploaded_file)
        context['url'] = '/media/'+uploaded_file
        text_list = uploaded_text.split(';')
        paths = pose(['/media/'+uploaded_file, text_list])

        # context['text'] = '/media/' + paths[0]
        context['neck'] = '/media/' + paths[1]
        context['left'] = '/media/' + paths[2]
        context['right'] = '/media/' + paths[3]
        context['video'] = '/media/' + paths[4]
        file = uploaded_file.split('.mp4')[0]
        context['neck_text'] = '/media/' + file + '_neck_pose.txt'
        context['left_text'] = '/media/' + file + '_left_upper_arm.txt'
        context['right_text'] = '/media/' + file + '_right_upper_arm.txt'

    return render(request, 'pose.html', context)


def analyze(request):
    context = {}
    if request.method == 'POST':
        neck_text = str(request.FILES['document1'])
        video_name = str(neck_text).split('_neck_pose.txt')[0]
        left_shoulder = str(request.FILES['document2'])
        right_shoulder = str(request.FILES['document3'])
        neck = request.POST.get('textfield1')
        shoulder = request.POST.get('textfield2')

        neck_text = '/media/' + neck_text
        left_shoulder = '/media/' + left_shoulder
        right_shoulder = '/media/' + right_shoulder
        neck = neck.split(';')
        shoulder = shoulder.split(';')

        video = [video_name, neck_text, left_shoulder, right_shoulder, neck, shoulder]

        # merge_shoulder_files(video)
        calculate_point(video)
        make_excel(video)

        context['excel'] = '/media/' + video[0]+'_analysis.xlsx'
        context['text'] = '/media/' + video[0] + '_pose_result.txt'

    return render(request, 'calculate.html', context)

