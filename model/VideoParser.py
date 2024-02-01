import os
import sys
import time

base_path = os.curdir
sys.path.insert(0, base_path)

from utils.process_video import *
#from utils.parse_hand_landmarker import *


raw_path = os.path.join(base_path, "raw_videos")
parsed_path = os.path.join(base_path, "parsed_data")
#print(parsed_path)

raw_paths = []
parsed_paths = []
new_dirs = set()

count = 0

def extension_remover(path):
    idx = path.rfind('.')
    return path[:idx:]



for root, d_names, f_names in os.walk(raw_path):
    for f in f_names:
        temp = os.path.join(root, f)
        removed = root[12::]
        parsed_paths.append(parsed_path + removed + "\\" + f)
        new_dirs.add(parsed_path + removed)
        raw_paths.append(temp)


for new_dir in new_dirs:
    try:
        os.mkdir(os.path.abspath(new_dir))
    except FileExistsError:
        continue

# print(raw_paths[0])
# frames = process_video(raw_paths[0])
# for frame in frames:
#     print(frame)

for i, video in enumerate(raw_paths):
    frames = process_video(video)
    print(frames.shape)
    np.save(extension_remover(parsed_paths[i]), frames)