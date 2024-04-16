##This model trains a Pytorch license plate image segmentation model

##Try to import Pytorch and check CUDA availability

try:
    import torch
    print("CUDA available? " + str(torch.cuda.is_available()))
except:
    print("Unable to import Pytorch, please install the proper version for your system...")

#Using roboflow, pull in license plate dataset
try:
    from roboflow import Roboflow
    print("Roboflow successfully imported")
except: 
    print("Unable to import Roboflow")
    
rf = Roboflow(api_key="zMt80qeo3jcFrWZ7aZC4")
project = rf.workspace("jodaryle-factor-gjkmr").project("license-plate-object-detection")
version = project.version(6)
dataset = version.download("coco")

#Since it is annotated in COCO, need to pull in additionall libraries

try:
    from torchvision.datasets import CocoDetection
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    from pycocotools.coco import COCO
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    print("Successfully imported necessary libraries")
    
except:
    print("Unable to import necessary libraries")

#Set Annotation path for training

coco_train_annotation = "License-Plate-Object-Detection-6/train/_annotations.coco.json"
coco_train = COCO(coco_train_annotation)
coco_train_path = "License-Plate-Object-Detection-6/train/" #actual image path

#Set Annotation path for validation

coco_validation_annotation = "License-Plate-Object-Detection-6/valid/_annotations.coco.json"
coco_validation = COCO(coco_validation_annotation)
coco_validation_path = "License-Plate-Object-Detection-6/valid/" #actual validation path

#Set Annotation path for testing

coco_testing_annotation = "License-Plate-Object-Detection-6/test/_annotations.coco.json"
coco_test = COCO(coco_testing_annotation)
coco_test_path = "License-Plate-Object-Detection-6/test/" #actual test path

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()])

#Create the datasets (which need COCO conversions for Pytorch)

train_dataset = CocoDetection(root=coco_train_path, annFile=coco_train_annotation, transform=transform)
validation_dataset = CocoDetection(root=coco_validation_path, annFile=coco_validation_annotation, transform=transform)
test_dataset = CocoDetection(root=coco_test_path, annFile=coco_testing_annotation, transform=transform)

#Create the dataloaders to start actually pulling data in

#Set an arbitrary batch size
batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

#Grab a random image from the training directory. Display its standard image and its ground truth
import random
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Fetch image IDs using COCO fn call
image_ids = coco_train.getImgIds()

random_image_id = random.choice(image_ids)

#Load image that correponds to the ID
image_info = coco_train.loadImgs(random_image_id)[0]
image_path = coco_train_path + image_info['file_name']
raw_image = io.imread(image_path)

#Now get annotation
annotation_id = coco_train.getAnnIds(imgIds=random_image_id)
annotation = coco_train.loadAnns(annotation_id)

#Show image
fig, ax = plt.subplots()
ax.imshow(raw_image)
ax.axis('off')
ax.set_title('Annotated Image')

#Display the annotation over the image
for ann in annotation:
        if ann['category_id'] == 1: #1 should correspond to the license plate. If it doesn't, try 0
            bbox = ann['bbox']
            bbox = [bbox[0], bbox[1], bbox[2], bbox[3]] #convert to x, y, width, height
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

#If "q" is pressed, close the plot nicely
def close(event):
        if event.key == 'q':
            plt.close()

#Connect keypress to figure
fig.canvas.mpl_connect('key_press_event', close)

plt.show()


#Time to start constructing the actual network
import torch.nn as nn
import torch.nn.functional as F

class PlateIdentifier(nn.Module):
        def __init__(self):
            super(PlateIdentifier, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #all of these images are 640x640, need to pad to maintain given a currnet convolve kernel of 16x16 pixels (3 ocrresponds to RGB)
            self.conv2 = nn.Conv2d(16, 31, 3, padding=1) #In channels have to correspond to previous layer out channel count
            self.pool = nn.MaxPool2d(2, 2) #2x2 pixel window that moves 2 pixels at a time
            self.fc1 = nn.Linear(31 * 160 * 160, 120)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x))) #Need two relus, two convolution layers
            x = x.view(-1, 31 * 160 * 160) #flattening to a 1D tensor. -1 tells Pytorch to infer the batch size dimension
            x = F.relu(self.fc1(x))
            return x

model = PlateIdentifier()
    
#Define our loss function - using binary cross-entropy because plate is either there or it isn't
loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        input, label = data

        #Convert labels and inputs to tensors explictly
        input = torch.tensor(input)
        label = torch.tensor(label)
        
        optimizer.zero_grad() #important to zero the gradient on every batch

        output = model(input)

        #Compute loss nad its gradient
        loss = loss_fn(output, label)
        loss.backward()

        #Adjust learning weights
        optimizer.step()

        #Gather & report data
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_time / 1000 #loss per batch
            print('  batch {} loss: {}'.format(i+1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_los, tb_x)
            running_loss = 0.
    
    return last_loss


#Per epoch data
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/plate_trainer_{}'.format(timestamp))
epoch_number = 0

epochs = 10

best_vloss = 1_000_000

for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    #Gradient tracking on, make a pass over the data!
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0

    #Switch to evaluation mode
    model.eval()

    #Reduce memory consumption by disabling gradient computation
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinpus)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss = vloss

    avg_vloss = running_vloss / (i+1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    #Log running loss averaged per batch for both the training AND validation
    writer.add_scalars('Training vs. Validation Loss',
                       { 'Training' : avg_loss, 'Validation' : avg_vloss },
                       epoch_number + 1)
    writer.flush()

    #Track best performance and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
    epoch_number += 1
          