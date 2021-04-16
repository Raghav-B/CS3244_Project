call ./yolov4-deepsort/Scripts/activate.bat
python3 object_tracker.py --video ./test_videos/final/night_street.mp4 --output ./outputs/night_street.mp4 --model yolov4
::python3 object_tracker.py --video ./test_videos/final/1.mp4 --model yolov4
