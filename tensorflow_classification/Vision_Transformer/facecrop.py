import sys
import cv2
import os

# ----------------------------------------------------------------------------
dataset_path=''

class Cropper(object):
    """Cropper"""
    def __init__(self):
        super(Cropper, self).__init__()
        self.detector  = Detector()

    @staticmethod
    def _bounding_rect(faces):
        top,    left  =  sys.maxsize,  sys.maxsize
        bottom, right = -sys.maxsize, -sys.maxsize
        for (x, y, w, h) in faces:
            if x < left:
                left = x
            if x+w > right:
                right = x+w
            if y < top:
                top = y
            if y+h > bottom:
                bottom = y+h
        return top, left, bottom, right

    def crop(self, img, target_width, target_height):
        original_height,  original_width  = img.shape[:2]
        faces = self.detector.detect_faces(img)
        if len(faces) == 0: # no detected faces
            target_center_x = original_width // 2
            target_center_y = original_height // 2
        else:
            top, left, bottom, right  = self._bounding_rect(faces)
            target_center_x = (left + right) // 2
            target_center_y = (top + bottom) // 2
        target_left = target_center_x - target_width // 2
        target_right = target_left + target_width
        target_top = target_center_y - target_height // 2
        target_bottom = target_top + target_height
        if target_top < 0:
            delta = abs(target_top)
            target_top += delta
            target_bottom += delta
            if target_bottom > original_height:
                target_bottom = original_height
        if target_left < 0:
            delta = abs(target_left)
            target_left += delta
            target_right += delta
            if target_right > original_width:
                target_right = original_width
        return img[target_top:target_bottom, target_left:target_right]

# ----------------------------------------------------------------------------

def face_crop(input_image, target_width, target_height):
    cropper = Cropper()

    target_image = cropper.crop(input_image, target_width, target_height)
    if target_image is None:
        print('Cropping failed.')

    return target_image

# ----------------------------------------------------------------------------

DEFAULT_DATA_FILENAME = os.path.join(
    os.path.split(os.path.realpath('__file__'))[0],
    dataset_path + '/haarcascade_frontalface_default.xml'
    )

class Detector(object):
    """Detector"""
    def __init__(self, data_filename = DEFAULT_DATA_FILENAME):
        super(Detector, self).__init__()
        self.face_cascade  = cv2.CascadeClassifier(data_filename)

    def detect_faces(self, img):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return faces
# ----------------------------------------------------------------------------