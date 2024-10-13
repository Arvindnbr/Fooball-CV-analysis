import cv2


def read_vid(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_vid(video_frames,output_path):
    format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, format, fps=25, frameSize=(video_frames[0].shape[1],video_frames[0].shape[0]))
    for frame in video_frames:
        out.write(frame)
    out.release()