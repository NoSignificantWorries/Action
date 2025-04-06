import os

import cv2


def video_to_image(input_video_path, output_path):
    print("Starting", input_video_path)

    cap = cv2.VideoCapture(input_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print(f"ERROR: Can't open this file {input_video_path}")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i in range(frame_count):
        ret, frame  = cap.read()
        if not ret:
            print(f"WARN: Can't reading frame {i}")
            break
        else:
            cv2.imwrite(os.path.join(output_path, f"frame_{i:03d}.jpg"), frame)

    cap.release()


path = "video_clips"
dataset_path = "dataset_img"

for class_name in os.listdir(path):
    class_dir_path = os.path.join(path, class_name)
    video_pathes = os.listdir(class_dir_path)
    save_dir = os.path.join(dataset_path, class_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for video in video_pathes:
        video_to_folder = video.replace(".mp4", "")
        video_to_image(os.path.join(class_dir_path, video), os.path.join(save_dir, video_to_folder))
