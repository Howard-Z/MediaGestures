import cv2
import sys
import argparse
import imageio
from time import sleep
from os import makedirs

'''
USAGE:
    Setup:
        Open cmd.exe, cd to the folder this file is in.

    Run:
        > python record_video.py <directory> <time> <count> [--RESOLUTION] [--FRAME_RATE]

    Parameters:
        directory: Path to a folder where output should be stored
            - If directory path contains spaces, place quotes around the section that does.
            - Entire directory path should NOT be in quotes.
            - For best results, use absolute paths, although relative paths may work.

        time: Length of each video, in seconds.
        
        count: Number of videos to record.
        
        --resolution [OPTIONAL]: Resolution of each video.
            - Format as "[num1]x[num2]".
            - Maximum resolution dependent on your hardware.
            - Defaults to 1280 x 720.
            
        --frame_rate [OPTIONAL]: Frame rate of each video.
            - Maximum frames per second dependent on your hardware.
            - Defaults to 30 fps.
    
    Return:
        - Outputs data to folder specified by <DIRECTORY>.
        - Videos are titled [num].avi, starting from 1 and counting up.
        - It is highly recommended to always write to an empty folder, or your data may be overwritten!
            - This is not enforced; you should check before running.

    Example Usage:
        python record_video.py C:/Video_Output 3 30 --resolution 640x480 --frame_rate 60
'''


def record_multiple(directory: str, seconds: int, count: int, resolution: tuple, fps: int):
    try:
        print("Initializing...")
        cap = cv2.VideoCapture(0)
        cap.set(3, resolution[0])
        cap.set(4, resolution[1])

        for i in range(1, count + 1):
            filepath = f'{directory}/{i}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, fps, resolution)

            if i < count + 1:
                sleep(2)
                print(f"Beginning next recording in 3 for video {i}...", end='\r')
                sleep(1)
                print(f"Beginning next recording in 2 for video {i}...", end='\r')
                sleep(1)
                print(f"Beginning next recording in 1 for video {i}...", end='\r')
                sleep(1)
                print("                                               ", end='\r')

            print(f"Recording video {i}")
            for _ in range(int(seconds * fps)):
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                # if frame is not None:
                    # cv2.imshow(f'Frame', frame)

            print(f"Video {i} completed.")
            out.release()

        cap.release()
        cv2.destroyAllWindows()
        print("Done recording.")
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        print("Error in recording:", str(e))
        
def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help="Directory to save the video file")
    parser.add_argument('seconds', type=int, help="Number of seconds to record")
    parser.add_argument('count', type=int, help="Number of videos to record")
    parser.add_argument('--resolution', type=str, default='1280x720', help='Video resolution (e.g., 1280x720)')
    parser.add_argument('--frame_rate', type=int, default=30, help="Frame rate of the video")
    
    args = parser.parse_args()

    seconds = args.seconds
    directory = args.directory
    count = args.count
    resolution = tuple(map(int, args.resolution.split('x')))
    frame_rate = args.frame_rate

    if resolution is None: resolution = (1280, 720)
    if frame_rate is None: frame_rate = 30

    makedirs(directory, exist_ok=True)

    record_multiple(directory, seconds, count, resolution, frame_rate)
    
if __name__ == '__main__':
    main()
    
