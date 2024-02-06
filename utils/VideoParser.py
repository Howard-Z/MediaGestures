import os
import sys
import time
from process_video import*
from get_root_dir import *
if sys.platform == 'darwin':
    slash = '/'
elif sys.platform == 'win32' or sys.platform == 'cygwin':
    slash = '\\'

#potentially breaking
#base_path = os.curdir
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
        #parsed_paths.append(parsed_path + removed + "\\" + f)
        parsed_paths.append(removed + slash + f)
        #parsed_paths.append(removed + "\\" + f)
        #new_dirs.add(parsed_path + removed)
        new_dirs.add(removed)
        raw_paths.append(temp)


for new_dir in new_dirs:
    try:
        os.mkdir(os.path.abspath(new_dir))
    except FileExistsError:
        continue


# for i, video in enumerate(raw_paths):
#     frames = process_video(video)
#     np.save(extension_remover(parsed_paths[i]), frames)
    
for i, video in enumerate(raw_paths):
    try:
        os.mkdir(os.path.abspath(extension_remover(parsed_paths[i])))
    except FileExistsError:
        pass
    if ".DS_Store" in video:
        continue
    frames = process_video(video)
    # print(frames.shape)
    for j, frame in enumerate(frames):
        # np.save(extension_remover(parsed_paths[i]) + '-' + str(j), frame)
        temp = os.path.join(os.getcwd(), extension_remover(parsed_paths[i]))
        if np.isnan(np.sum(frame[0])):
            # print("skipped left!")
            # print(frame[0])
            pass
        else:
            #np.save(temp + "/"+ str(j) + "-l", frame[0])
            np.save(temp + slash + str(j) + "-l", frame[0])
            # print("Left")
            # print(frame[0])

        if np.isnan(np.sum(frame[1])):
            # print("skipped right!")
            # print(frame[1])
            pass
        else:
            # print(temp)
            #np.save(temp + "/" + str(j) + "-r", frame[1])
            np.save(temp + slash + str(j) + "-r", frame[0])
            # print("right")
            # print(frame[1])