import glob
import cv2
import os


RA_Video_dir_real = ' '
RA_Video_dir_fake = ' '
true_store_dir =  ' '
fake_store_dir =  ' '

real_count = 0
fake_count = 0
# 获取文件夹下所有的mov后缀
frame_count = 0
real_list_mov = glob.glob(os.path.join(RA_Video_dir_real, '*.mov'))
for mov in real_list_mov:
  cv1 = cv2.VideoCapture(mov)
  flag, frame = cv1.read()
# Calculating total frames
  while (flag):
    ret, frame = cv1.read()
    if ret is False:
      break
    frame_count = frame_count + 1
  cv1.release()
print(frame_count)
print(real_list_mov)
print(len(real_list_mov))

gap = 20
c = 0
real_list_mov = glob.glob(os.path.join(RA_Video_dir_real, '*.mov'))
n = 0
for mov in real_list_mov:
  n = n + 1
  cv1 = cv2.VideoCapture(mov)
  flag, frame = cv1.read()
  while (flag):
    flag, frame = cv1.read()
    if flag == True:
      c = c + 1
      if (c % gap == 0):
          # target_image = face_crop(frame, 600, 800)
          cv2.imwrite(true_store_dir + 'real' + '.' + str(real_count) + '.jpg', frame)
          real_count = real_count + 1
          print('完成第'+str(real_count)+'张图片的保存')
  # cv1.release()
  print("real images:"+str(real_count))
print(n)
print("real images:"+str(real_count))
#
# real_count = 0
# fake_count = 0
# # 获取文件夹下所有的mov后缀
# frame_count = 0
# fake_list_mov = glob.glob(os.path.join('fixed/hand', '*.mov'))
# for mov in fake_list_mov:
#   cv1 = cv2.VideoCapture(mov)
#   flag, frame = cv1.read()
# # Calculating total frames
#   while (flag):
#     ret, frame = cv1.read()
#     if ret is False:
#       break
#     frame_count = frame_count + 1
#   cv1.release()
# print(frame_count)
# print(fake_list_mov)
# print(len(fake_list_mov))
#
# gap = 40
# c = 0
# fake_list_mov = glob.glob(os.path.join(' ', '*.mov'))
# n = 0
# for mov in fake_list_mov:
#   n = n + 1
#   cv1 = cv2.VideoCapture(mov)
#   flag, frame = cv1.read()
#   while (flag):
#     flag, frame = cv1.read()
#     if flag == True:
#       c = c + 1
#       if (c % gap == 0):
#           # target_image = face_crop(frame, 600, 800)
#           cv2.imwrite(fake_store_dir + 'fake' + '.' + str(fake_count) + '.jpg', frame)
#           fake_count = fake_count + 1
#           print('完成第'+str(fake_count)+'张图片的保存')
#   # cv1.release()
#   print("fake images:"+str(fake_count))
# print(n)
# print("fake images:"+str(fake_count))