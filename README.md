# MediaGestures
UCLA CS 188 Final Project

There's cuda versions of pytorch, if you need that use the command: 

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` 

## Data Collection
The data collection was supposed to be done by running `./helpers/record_video.py`.  However, it turns out that python-opencv (cv2) struggles to write frames to file at more than 10 frames per second, and so do many other libraries.  This is apparently well documented on the internet - many people have had the same problem.

#### `./helpers/record_video.py` is deprecated.  Don't use it!

To avoid having to deal with this issue, I decided it would be easier if a third party recording software was used instead, which adds some dependencies.

**Dependencies:**
* **ffmpeg** (not the python versions, but the .exe file you get from [here](https://ffmpeg.org/download.html))
* **[OBS Studio](https://obsproject.com/download)** or some other recording software of your choice.

Note: These may take a while to install.

### Recording
The script `./helpers/clicker.py` is essentially an autoclicker that clicks in the pattern (click, wait 5 seconds, click, wait 4 seconds, repeat).  This is a script to be run in command prompt.  First, change directory into the folder with the script. Run:

`python clicker.py <num_videos_to_record>`

Once it prints `Initializing`, go to OBS studio (or any other recording software), and hover your mouse over the record start/stop button.  A three second timer will start (in the terminal), and when it expires, the mouse will click on the start recording button.  At this point, do the gesture you are training for.  **Do not take up the entire time of the video or your gesture may be trimmed during standardizing.  The gesture should happen in three seconds, maximum.**  After about four seconds, the recording will be stopped, and you will have a couple seconds before the next recording starts.  This process will repeat <num_videos_to_record> times.  `Done recording.` will be printed in the terminal after the last video is finished recording.

OBS Studio or your recording software should be configured as the following:
* 1920 x 1080 output resolution (if your hardware does not support this, use whatever the maximum is)
* 30 Frames per second
* Muted audio
* .mp4 file output

### Video Standardizing
Once recording is done, gather all the video files into the same folder.  Make sure that the folder does not contain any other files, other than the recorded videos.  Make a copy of `./helpers/trim_video.py`, into the folder with the videos, and double click to run it.  It will create a new folder titled `trimmed_videos` that contains a copy of all the videos in the folder, trimmed down to exactly three seconds (90 frames total).  Save these videos as training data.  You can delete the unprocessed videos.

## Data Processing
Data processing is done by a couple of scripts located in the `./model` directory

**Dependencies:**
Data processing depends on the following python files:
* `./utils/VideoParser.py`
* `./utils/process_video.py`
* `./model/HandLandMarkDataset.py`

### Running
After you have recorded your training videos. Move them under the directory `./raw_videos` under their correct class folder. For example, if you just recoreded a pinch gesture called "pinch1.mp4", then you would move it to `./raw_videos/pinch/pinch1.mp4`

Once you have moved all of the videos, run `VideoParser.py`. This will take all the videos, load them, and then pass them to `process_video.py` for processing each frame. They will then get saved as a numpy array to disk with a .npy file.

Note: the directory structure of the `unsorted_data` folder will mirror the folder structure of `raw_videos`. Each video will get their own folder and each frame will be saved in one or two files depending on how many hands are present.

The naming scheme of data files are "frameNum-handedness.npy"

After the parser is finished you can proceed to the dataset.

Enter the `unsorted_data` folder and sort them into training/validation sets in the `parsed_data` folder as you wish.  This is not done automatically.  Refer to the sample data (explained below) as an example.

### Loading the dataset
The `HandLandMarkDataset.py` file contains the necessary class derivation to be a valid dataset class for pytorch.

### Sample Data
Some sample data is provided in the repository in `parsed_data.zip`.  Unzip the folder to use it.  This data has been manually divided into training and validation samples.  Data is saved as .npy files (ndarrays in file form), and are generated from videos using `VideoParser.py`.  Raw videos are not included.