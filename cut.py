import os

import cv2

def cut_video(input_video_path, output_folder, timestamps, video_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print(f"ERROR: Can't open this file {input_video_path}")
        return

    for i, (start_time, end_time) in enumerate(timestamps):
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        if start_frame < 0:
            start_frame = 0
        if end_frame > frame_count:
            end_frame = frame_count

        output_video_folder = os.path.join(output_folder, f"{video_name}_clip_{i:3d}")
        if not os.path.exists(output_video_folder):
            os.makedirs(output_video_folder)
        
        output_file_path = os.path.join(output_folder, f"{video_name}_clip_{i:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print(f"WARN: Can't reading frame {frame_num}")
                break
            else:
                out.write(frame)

        out.release()
        print(f"Element {i} saved in {output_video_folder}")

    cap.release()


f = (lambda x: tuple(map(int, x.split(":"))))

if __name__ == "__main__":
    video_open_dir = "/home/dmitrykarp/Videos/Dataset"
    annotating_file = "violin_annotations.txt"
    
    input_videos = {}
    with open(os.path.join(video_open_dir, annotating_file), "r") as file:
        lines = list(map(lambda x: x[:-1], file.readlines()))
        m = int(lines[0])
        i = 1
        for j in range(m):
            n = int(lines[i + 1])
            time_stamps = lines[i + 2:i + n + 2]
            data = []
            for stamp in time_stamps:
                stamp = stamp.split(" ")
                t1 = tuple(map(int, stamp[0].split(":")))
                t2 = tuple(map(int, stamp[1].split(":")))
                t1 = t1[0] * 60 + t1[1]
                t2 = t2[0] * 60 + t2[1]
                data.append((t1, t2))

            input_videos[lines[i]] = data
            i = i + n + 2
    
    for video_name in input_videos.keys():
        input_video = os.path.join(video_open_dir, video_name)
        output_directory = "video_clips/violin"

        cut_video(input_video, output_directory, input_videos[video_name], video_name)
