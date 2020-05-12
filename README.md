# fare avoidance detection

## People Tracker

![](passenger_fare_detection.gif)

### How to run
To Run with the test example videos
```bash
py3 main.py --input videos/example_01.mp4 --output output/output_01.avi
```

To Run with webcam
```bash
py3 main.py --output output/output_01.avi
```

### Installation
Set up dependencies by
```bash
cd fare_evasion && pipenv install
```

### Running backend
```bash
cd fydp-backend && pipenv run python blacka.py
```

### Phone Camera Setup
ANDROID:<br />
Install: "DroidCam" from playStore<br />
Install the DroidCam Client (Windows) <br />
http://www.dev47apps.com/droidcam/windows/

1. Start mobile app, and get the IP and port number
2. Within the DroidCam client, insert the mobile device IP and Port number
3. Run people_tracker and it will use your phone as the webcam

### Faster RCNN Setup
prototxt and pre-trained models downloaded from 
https://docs.openvinotoolkit.org/2018_R5/_samples_object_detection_demo_README.html


## Acknowledgments
We used the following repositories and resources. Thank you for helping
* https://github.com/nwojke/deep_sort
* https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
* https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
