import os
import subprocess

def trim_videos_to_n_frames(input_folder, output_folder, n_frames):
    try:
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # List all .mp4 files in the input folder
        video_files = [file for file in os.listdir(input_folder) if file.endswith(".mp4")]

        for video_file in video_files:
            input_filepath = os.path.join(input_folder, video_file)
            output_filepath = os.path.join(output_folder, f"trimmed_{video_file}")

            # Use ffmpeg to trim the video
            ffmpeg_command = [
                'ffmpeg',
                '-i', input_filepath,
                '-vf', f'select=lt(n\,{n_frames})',
                '-q:v', '2',
                '-y',
                output_filepath
            ]

            subprocess.run(ffmpeg_command, check=True)

            print(f"{video_file} trimmed to {n_frames} frames and saved to {output_filepath}")

    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    input_folder = os.getcwd()  # Current working directory
    output_folder = os.path.join(input_folder, "trimmed_videos")
    n_frames = 90  # Specify the number of frames you want

    trim_videos_to_n_frames(input_folder, output_folder, n_frames)