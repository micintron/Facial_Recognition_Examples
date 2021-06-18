# Requirements 
Python v3.7+

The following scripts leverage OpenCV + face_recognition libraries. 
## Live Stream Recognition
This demo showcases the capability of recognizing faces through live stream. The script leverages the webcam on the computer and can detect multiple faces at a time. 
### Instructions

Before you start, please install all dependencies: 

```pip install -r requirements.txt```

In order to run the Live Feed detection: 

```python live_feed_detection.py```

## Video Recognition
This demo showcases providing a video and an image of the face to recognize. The script will parse the video frame by frame and look for matches based on the provided image. 

Note: Due to the nature of the CI Challenge, we were not able to fully optimize. A one second clip still contains ~30 frames, and each frame takes approximately 1 - 2 seconds to render. While this showcases what is functionally possible through this library, scalability will require more research. 

### Instructions
Before you start, please install all dependencies:

```pip install -r requirements.txt```


Make sure the following is in the root folder:
- ```input_video.mp4``` - this is the video that will be searched for faces
- ```input_image.jpeg``` - this is the image that will be matched in the video

In order to run the Live Feed detection: 

```python video_detection.py```

The output video will be labeled `output_video.mp4`


You can find already identified videos in the `rendered_examples` folder.
- obama.mp4 (1 sec video) -> obama_output.mp4
- obama2.mp4 (10 sec video) -> obama_2_output.mp4
