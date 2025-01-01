import json

from deploy import predictor

# Input video file (stored in S3 or uploaded locally)
input_video_path = "s3://footballanalysisbucket/input_video.mp4"
output_video_path = "s3://footballanalysisbucket/output_video.mp4"

# Prepare the payload
payload = json.dumps({
    "input_video": input_video_path,
    "output_video": output_video_path
})

# Make the request
predictor.predict(payload)
