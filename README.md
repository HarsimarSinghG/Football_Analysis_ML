# Football_Analysis_ML
The training dataset is under license of Roboflow and it can be obtained at the following link:
https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1

I have provided only a single sample for the training dataset in this repo. However, if you could please
contact me at my email [simar@mail.utoronto.ca](url), I can provide you access to my private repo that contains
the trained model weights and final video for test dataset.

This project involves the use of open Computer Vision library to detect football matches in real-time tracking
that includes tracking players, football even out of frames. Also, it includes the speed and distance traveled by players.
Interpolation was used for the position of the ball to achieve tracking in all the frames.
To calculate speed and distance in meters rather than pixels, the perspective transform module of cv2 was used.

Issues to fix: Deployment on AWS SageMaker on a private endpoint to make inference is not working as required.

More features to be added: Detecting time frames for events of the ball going offside, requires keypoint extraction of the court.

# Example of dataset
![2e57b9_1_8_png rf abd7f2df70d4bf0ab5e307b0f1c3d75b](https://github.com/user-attachments/assets/1ed0658d-a473-4e3f-8043-d94e8eecf43a)
2 0.21844791666666669 0.37982407407407404 0.02145312499999997 0.053870370370370325
1 0.07291666666666667 0.31851851851851853 0.010416666666666666 0.04814814814814815
3 0.48151041666666666 0.38842592592592595 0.009895833333333333 0.05648148148148148
0 0.24934375 0.3975740740740741 0.005630208333333305 0.009111111111111141

Here the first number includes class id of each detected object and the following floats represent the 
coordinates of the bounding boxes of each object.

class id:
0 - ball
1 - goalkeeper
2 - player
3 - referee

# Sample frame in the video detection of test dataset
<img width="531" alt="Screenshot 2024-12-31 at 7 12 36 PM" src="https://github.com/user-attachments/assets/634ccb47-1b16-4de9-97a9-2217ffebb1e2" />

Please note that the following video was used as a reference for the annotator object to achieve better GUI on the final video.
https://www.youtube.com/watch?v=neBZ6huolkg&t=15408s

