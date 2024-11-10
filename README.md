# Facial Emotion Detection

## Context:
This modified use case is designed to inspire individuals to explore the application of AI across various fields. As a food science student, I have adapted this case study to showcase how AI can be leveraged in science exhibitions, particularly for engaging early-age students. The model and code were originally sourced from [spmallick/learnopencv](https://github.com/spmallick/learnopencv/tree/master/Facial-Emotion-Recognition), and I have further customized it to create an executable (.exe) file.

I would like to express my gratitude to [OpenCV University](https://opencv.org/university/) for offering valuable resources and courses. Additionally, I would like to acknowledge the other platforms and communities from which I have learned and further refined the base code.

1. This is a good read [Build Your Own Face Recognition Tool With Python](https://realpython.com/face-recognition-with-python/),

2. Emotion Detection with FER
a. [Emotions Detect of face | Using Fer , OpenCV](https://youtu.be/MWskZw791d0?si=ouue7aqqhD2vLj3c)
b. https://www.edlitera.com/en/blog/posts/emotion-detection-in-video
c. https://youtu.be/xv3G5sIx2co?si=83HDU7KXZqpgNdcn
d. https://youtu.be/Bb4Wvl57LIk?si=WldmTYibB1QGvt13 
e. [Emotion Detection using CNN](https://youtu.be/UHdrxHPRBng?si=Z-9XDwTIskU5litK) 1 (FER)
f. [Emotion Detection using OpenCV & Python by Edureka](https://youtu.be/G1Uhs6NVi-M?si=t2iHXmF4DOZNa3qb)
g. https://youtu.be/aoCIoumbWQY?si=htK8p3ohAC5-9y7l 2 (FER)

3. Hugging Face
https://youtu.be/QEaBAZQCtwE?si=Ey30n12gPQrbF9OY

## How to use:

The folder structure looks like this,
emotion_detection/
├── main.py
├── build_exe.py
├── requirements.txt
└── models/
    ├── emotion-ferplus-8.onnx
    └── RFB-320/
        ├── RFB-320.caffemodel
        └── RFB-320.prototxt

1. create your venv and install packages,
```
python -m venv ./venv

# Activate it
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```
2. generate the .exe

```
# running build script
python build_exe.py
```
this will generate a standalone .exe file at 
emotion_detection/
└── dist/
    ├── EmotionDetection.exe

Thank you.