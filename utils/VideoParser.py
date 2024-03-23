import os
import sys
import time
from process_video import*
from get_root_dir import *
if sys.platform == 'darwin':
    slash = '/'
elif sys.platform == 'win32' or sys.platform == 'cygwin':
    slash = '\\'

base_path = get_root_dir()

raw_path = os.path.join(base_path, "raw_videos")
parsed_path = os.path.join(base_path, "parsed_data")

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
        removed = root.replace("raw_videos", "parsed_data")
        parsed_paths.append(removed + slash + f)
        new_dirs.add(removed)
        raw_paths.append(temp)


for new_dir in new_dirs:
    try:
        os.mkdir(os.path.abspath(new_dir))
    except FileExistsError:
        continue
    
for i, video in enumerate(raw_paths):
    try:
        os.mkdir(os.path.abspath(extension_remover(parsed_paths[i])))
    except FileExistsError:
        pass
    if ".DS_Store" in video:
        continue
    frames = process_video(video)
    for j, frame in enumerate(frames):
        temp = os.path.join(os.getcwd(), extension_remover(parsed_paths[i]))
        
        # left hand
        if not np.isnan(np.sum(frame[0])): 
            np.save(temp + slash + str(j) + "-l", frame[0])

        # right hand
        if not np.isnan(np.sum(frame[1])): 
            np.save(temp + slash + str(j) + "-r", frame[1])
