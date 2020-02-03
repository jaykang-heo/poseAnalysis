import os
import xlsxwriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = BASE_DIR.replace('\\'[0], '/')


def merge_shoulder_files(video):
    orig_right = open(path+'/'+video[3], 'r')
    orig_left = open(path+'/'+video[2], 'r')
    new = open(path+'/'+video[0]+'_shoulder.txt', 'w')
    for left, right in zip(orig_left.readlines(), orig_right.readlines()):
        left = left.split('\n')[0]
        right = right.split('\n')[0]
        if left == 'nan' and right != 'nan':
            print('hello')
            new.write(right+'\n')
        elif left != 'nan' and right == 'nan':
            new.write(left+'\n')
        elif left != 'nan' and right != 'nan':
            num = (float(left)+float(right))/2
            new.write(str(num)+'\n')
        else:
            new.write(left+'\n')
    new.close()


def calculate_point(video):
    neck_text = open(path+'/'+video[1], 'r')
    upper1_text = open(path+'/'+video[2], 'r')
    upper2_text = open(path+'/'+video[3], 'r')
    pose_result = open(path + '/' + '/media/' + video[0] + '_pose_result.txt', 'w')
    points = []
    count = 0
    nan = 0
    # for angles in result:
    neck_list = []
    left_upper_arm_list = []
    right_upper_arm_list = []
    ################################################################
    ################################################################
    ################################################################
    # neutral upper bound, neutral lower bound, flex upper bound, ext lower bound
    neck_neutral = video[4]
    # shoulder mild range, severe range
    shoulder = video[5]
    ################################################################
    ################################################################
    ################################################################
    for neck, upper1, upper2 in zip(neck_text.readlines(), upper1_text.readlines(), upper2_text.readlines()):
        if neck.strip('\n') != 'nan':
            neck = float(neck.strip('\n'))
            neck_neutral[0] = int(neck_neutral[0])
            neck_neutral[1] = int(neck_neutral[1])
            upper1 = float(upper1.strip('\n'))
            upper2 = float(upper2.strip('\n'))
            shoulder[0] = int(shoulder[0])
            shoulder[1] = int(shoulder[1])
            if neck_neutral[0] > neck > neck_neutral[1]:
                neck_number = 1
                neck_list.append(neck_number)
            elif neck_neutral[1] >= neck:
                neck_number = 2
                neck_list.append(neck_number)
            elif neck_neutral[0] <= neck:
                neck_number = 3
                neck_list.append(neck_number)
            else:
                pass

            if shoulder[0] > upper1:
                left_upper_arm_number = 1
                left_upper_arm_list.append(left_upper_arm_number)
            elif shoulder[1] > upper1 >= shoulder[0]:
                left_upper_arm_number = 2
                left_upper_arm_list.append(left_upper_arm_number)
            elif upper1 > shoulder[1]:
                left_upper_arm_number = 3
                left_upper_arm_list.append(left_upper_arm_number)
            else:
                pass
                # left_upper_arm_number = 1
                # left_upper_arm.append(left_upper_arm_number)

            if shoulder[0] > upper2:
                right_upper_arm_number = 1
                right_upper_arm_list.append(right_upper_arm_number)
            elif shoulder[1] > upper2 >= shoulder[0]:
                right_upper_arm_number = 2
                right_upper_arm_list.append(right_upper_arm_number)
            elif upper2 >= shoulder[1]:
                right_upper_arm_number = 3
                right_upper_arm_list.append(right_upper_arm_number)
        else:
            nan += 1
        count += 1
    points.append([neck_list, left_upper_arm_list, right_upper_arm_list, count])

    neck_neutral = points[0][0].count(1)
    neck_flex = points[0][0].count(2)
    neck_ext = points[0][0].count(3)
    neck_miss = nan
    neck_total = count

    upper_arm_neutral = points[0][1].count(1)
    upper_arm_flex = points[0][1].count(2)
    upper_arm_ext = points[0][1].count(3)
    upper_arm_miss = nan

    upper_arm1_neutral = points[0][2].count(1)
    upper_arm1_flex = points[0][2].count(2)
    upper_arm1_ext = points[0][2].count(3)
    upper_arm1_miss = nan

    pose_total = ''
    pose_total += 'Total Frame: ' + str(count) + '\n'
    neck = str(neck_neutral * 100 / neck_total) + ' / ' + str(neck_flex * 100 / neck_total) + ' / ' + str(
        neck_ext * 100 / neck_total) + ' / ' + str(neck_miss * 100 / neck_total)
    upper_arm = str(upper_arm_neutral * 100 / neck_total) + ' / ' + str(
        upper_arm_flex * 100 / neck_total) + ' / ' + str(upper_arm_ext * 100 / neck_total) + ' / ' + str(
        upper_arm_miss * 100 / neck_total)
    upper_arm1 = str(upper_arm1_neutral * 100 / neck_total) + ' / ' + str(
        upper_arm1_flex * 100 / neck_total) + ' / ' + str(upper_arm1_ext * 100 / neck_total) + ' / ' + str(
        upper_arm1_miss * 100 / neck_total)
    pose_total += 'Neck: ' + neck + '\n' + 'Upper Arm(Shoulder): ' + upper_arm + '\n' + 'Upper Arm1(Shoulder): ' + upper_arm1 + '\n'
    pose_result.write(pose_total)
    pose_result.close()


def make_excel(video):

    path = BASE_DIR.replace('\\'[0], '/')
    workbook = xlsxwriter.Workbook(path + '/media/' + video[0]+'_analysis.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'video_name')
    worksheet.write('B1', 'total_frame')
    worksheet.write('C1', 'not_detected')
    worksheet.write(0, 3, 'Neck')
    worksheet.write(0, 11, 'Left Shoulder')
    worksheet.write(0, 19, 'Right Shoulder')
    worksheet.write(1, 3, 'neutral')
    worksheet.write(1, 5, 'flex')
    worksheet.write(1, 7, 'ext')
    worksheet.write(1, 9, 'miss')
    worksheet.write(1, 11, 'neutral')
    worksheet.write(1, 13, 'mild')
    worksheet.write(1, 15, 'severe')
    worksheet.write(1, 17, 'miss')
    worksheet.write(1, 19, 'neutral')
    worksheet.write(1, 21, 'mild')
    worksheet.write(1, 23, 'severe')
    worksheet.write(1, 25, 'miss')
    i = 2

    f = open(path+'/'+ '/media/' + video[0]+'_pose_result.txt', 'r')
    total = 0
    worksheet.write(i, 0, video[0])
    for line in f.readlines():
        line = line.split(':')
        if line[0] == 'Total Frame':
            line[1] = line[1].split('\n')[0]
            worksheet.write(i, 1, line[1])
        elif line[0] == 'Box Not Detected':
            line[1] = line[1].split('\n')[0]
            total += int(line[1])
        elif line[0] == 'Pose Not Detected (Box Detected)':
            line[1] = line[1].split('\n')[0]
            total += int(line[1])
            worksheet.write(i, 2, total)
        elif line[0] == 'Neck':
            line[1] = line[1].split('\n')[0]
            word = line[1].split('/')
            worksheet.write(i, 3, round(float(word[0].strip()), 3))
            worksheet.write(i, 5, round(float(word[1].strip()), 3))
            worksheet.write(i, 7, round(float(word[2].strip()), 3))
            worksheet.write(i, 9, round(float(word[3].strip()), 3))
        elif line[0] == 'Upper Arm(Shoulder)':
            line[1] = line[1].split('\n')[0]
            word = line[1].split('/')
            worksheet.write(i, 11, round(float(word[0].strip()), 3))
            worksheet.write(i, 13, round(float(word[1].strip()), 3))
            worksheet.write(i, 15, round(float(word[2].strip()), 3))
            worksheet.write(i, 17, round(float(word[3].strip()), 3))
        elif line[0] == 'Upper Arm1(Shoulder)':
            line[1] = line[1].split('\n')[0]
            word = line[1].split('/')
            worksheet.write(i, 19, round(float(word[0].strip()), 3))
            worksheet.write(i, 21, round(float(word[1].strip()), 3))
            worksheet.write(i, 23, round(float(word[2].strip()), 3))
            worksheet.write(i, 25, round(float(word[3].strip()), 3))
    i += 1
    workbook.close()

