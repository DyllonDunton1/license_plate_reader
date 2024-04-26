from ultralytics import YOLO
from roboflow import Roboflow


#Import yolov8 style annotations
rf = Roboflow(api_key="zMt80qeo3jcFrWZ7aZC4")
project = rf.workspace("cj-santos-e0nrn").project("license-plate-detection-3lgox")
version = project.version(3)
dataset = version.download("yolov8")

##Start YOLO training

#Load a base model
model = YOLO('yolov8n.pt')

#Train!
results = model.train(
    data='data.yaml',
    imgsz=640,
    epochs=25,
    batch=8,
    device=0,
    optimizer='Adam',
    verbose=True,
    cos_lr=True,
    dropout=0.25,
    plots=True,
    name='yolov8_plate'
)

"""
#Import loss graphing libraries
import matplotlib.pyplot as plt

training_accuracy = results['metrics']['train']['P']
val_accuracy = results['metrics']['val']['P']

epochs = range(1, len(training_accuracy) + 1)

plt.plot(epochs, training_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
"""
plt.show()
