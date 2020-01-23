import os
import cv2
import numpy as np
from proj_pipeline import *
from project_utilities import load_images_from_dir

# The following video processing code is copied and modified from my problem set #3 submission
VIDEO_DIR = "my_video"
if not os.path.isdir(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)


def create_video(video_name, model, windowsize):

    video = os.path.join(VIDEO_DIR, video_name)
    image_gen = video_frame_generator(video)

    fps = 15
    scale_values = [(1, 8), (1.25, 8), (1.5, 8), (1.75, 8), (2, 16)]
    frame_num = 1
    image = image_gen.__next__()
    image = cv2.resize(image, None, fx=.25, fy=.25)
    h, w, d = image.shape
    print('image.shape', image.shape)
    out_path = "{}/output_{}".format(VIDEO_DIR, video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    while image is not None:

        print("Processing fame {}".format(frame_num))

        image = cv2.resize(image, (w, h))
        if frame_num % 2 == 0:
            image = proj_pipeline(model, image, scale_values, windowsize)

            out_str = "video.{}.png".format(frame_num)
            save_image(out_str, image)
            video_out.write(image)

        image = image_gen.__next__()

        frame_num += 1

    video_out.release()


def read_dir_write_video(video_name, vid_dir='.', img_size=(1024, 512)):
    print('Getting image files...')
    imgs = load_images_from_dir(data_dir=vid_dir, size=img_size)
    print('Opening output video...')
    out_path = "{}/output_{}".format(vid_dir, video_name)
    video_out = mp4_video_writer(out_path, img_size, 15)
    print('Writing imgs to video..', len(imgs))
    for img in imgs:
        video_out.write(img)

    video_out.release()

def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(VIDEO_DIR, filename), image)


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None

# End of copied P3 code