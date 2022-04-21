import cv2
from facecrop import face_crop
FASD_Video_dir =  '/CASIA_FASD/train_release/'

true_store_dir = '/True_Face/'
fake_store_dir = '/Fake_Face/'

total = 0
for j in range(1,21):
    for i in range(1, 13):  # Each person has 12 videos
        vc = cv2.VideoCapture(FASD_Video_dir + str(j) + '/' + str(i) + '.avi')
        flag, frame = vc.read()
        # Calculating total frames
        frame_count = 0
        while (flag):
            ret, frame = vc.read()
            if ret is False:
                break
            frame_count = frame_count + 1
        print(frame_count)
        total = total+frame_count
        vc.release()
print('images:',total)

real_count = 0
fake_count = 0
for j in range(1,21):
    for i in range(1, 13):  # Each person has 12 videos
        vc = cv2.VideoCapture(FASD_Video_dir + str(j) + '/' + str(i) + '.avi')
        flag, frame = vc.read()
        # Calculating total frames
        frame_count = 0
        while (flag):
            ret, frame = vc.read()
            if ret is False:
                break
            frame_count = frame_count + 1
        print(frame_count)
        vc.release()

        gap = frame_count // 20
        c = 1

        vc = cv2.VideoCapture(FASD_Video_dir + str(j) + '/' + str(i) + '.avi')
        flag, frame = vc.read()
        while (flag):
            flag, frame = vc.read()
            if (flag == 0):
                break
            if (c % gap == 0):
                if (i <= 3):
                    if (i <= 2):
                        target_image = face_crop(frame, 240, 240)
                    else:
                        target_image = face_crop(frame, 600, 800)

                    cv2.imwrite(true_store_dir + 'real' + '.' + str(real_count) + '.png', target_image)
                    real_count = real_count + 1
                else:
                    if (i <= 9):
                        target_image = face_crop(frame, 240, 240)
                    else:
                        target_image = face_crop(frame, 600, 800)

                    cv2.imwrite(fake_store_dir + 'fake' + '.' + str(fake_count) + '.png', target_image)
                    fake_count = fake_count + 1
            c = c + 1
        cv2.waitKey(1)
    print("real images:"+str(real_count))
    print("fake images:"+str(fake_count))