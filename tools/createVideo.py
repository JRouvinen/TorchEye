import cv2
import os
import numpy

image_folder = 'C:/Users/Kimiwaha/AI/datasets/speed_test_1'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    stream = open(os.path.join(image_folder, image), "rb")
    bytes = bytearray(stream.read())
    numpyarray = numpy.asarray(bytes, dtype=numpy.uint8)
    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    video.write(bgrImage)
    #video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()