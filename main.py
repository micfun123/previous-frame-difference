import cv2 as cv
import numpy as np

video = cv.VideoCapture('rick.mp4')

frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(video.get(cv.CAP_PROP_FPS))
frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))

def process_frame(frame1, frame2):
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros_like(frame1)
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

frames = []
ret, prev_frame = video.read()
frame_number = 1
while video.isOpened():
    ret, next_frame = video.read()
    if ret:
        frames.append(process_frame(prev_frame, next_frame))
        prev_frame = next_frame
        frame_number += 1
        progress = round(frame_number / frame_count * 100, 2)
        progress_bar = '#' * int(progress) + ' ' * (100 - int(progress))
        print(f"Processing: {progress} {progress_bar}|", end='\r')
    else:
        break

video.release()

# Saving frames as a video with the original frame rate
out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
for i, frame in enumerate(frames):
    out.write(frame.astype('uint8'))
    progress = round((i + 1) / len(frames) * 100, 2)
    progress_bar = '#' * int(progress) + ' ' * (100 - int(progress))
    print(f"Writing: {progress} {progress_bar}|", end='\r')

out.release()
