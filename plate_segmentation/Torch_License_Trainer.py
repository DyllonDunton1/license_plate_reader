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
    import os
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    from pycocotools.coco import COCO as CocoDetection
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    import cv2 as cv
    import numpy as np
    print("Successfully imported necessary libraries")
    
except:
    print("Unable to import necessary libraries")

#Since the data we are using is annotated beyond what we want, need to pull out the proper annotations
#To do this, make a custom COCO dataset loader
class CustomCocoDetection(Dataset):
    def __init__(self, root, annotationFile, transform=None, target_category_id=1, image_size=(640, 640)):
        self.root = root
        self.coco = CocoDetection(annotationFile)
        self.transform = transform
        self.target_category_id = target_category_id
        self.image_size = image_size
        self.filtered_annotations = self.filter_annotations()
        
    def filter_annotations(self):
        filtered_annotations = []
        for ann in self.coco.dataset['annotations']:
            if ann['category_id'] == self.target_category_id:
                filtered_annotations.append({  
                    'image_id' : ann['image_id'],
                    'bbox' : ann['bbox'],
                    'segmentation' : ann.get('segmentation', None), #Get segmentation data if it is available
                    'label' : self.target_category_id
                    })
        return filtered_annotations
        
    def __getitem__(self, index):
        annotation = self.filtered_annotations[index]
        
        #Load an image
        img_info = self.coco.loadImgs(annotation['image_id'])[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        #Get bounding box coordinates
        bbox = annotation['bbox']
        bbox = [int(coord) for coord in bbox] #make sure it is integer coords only
        
        #Make the segmentation mask
        mask = np.zeros(self.image_size, dtype=np.uint8)
        #print("Segmentation Mask Size:", mask.shape) #Check size
        if annotation['segmentation']: #Use segmentation data if possible
            seg_points = np.array(annotation['segmentation'][0]).reshape((-1 ,2))
            seg_points = np.round(seg_points).astype(np.int32)
            cv.fillPoly(mask, [seg_points], 1) #Fill license plate region with 1
        
        if self.transform is not None:
            img = self.transform(img)
            
        label = annotation['label']
        segmentation_label = mask
        segmentation_label = np.expand_dims(segmentation_label, axis=0) #Add a dimension for batch
        return img, label, segmentation_label
    
    def __len__(self):
        return len(self.filtered_annotations)


transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()])

#Set Annotation path for training
coco_train_annotation = "License-Plate-Object-Detection-6/train/_annotations.coco.json"
coco_train = CocoDetection(coco_train_annotation)
coco_train_path = "License-Plate-Object-Detection-6/train/" #actual image path

#Set Annotation path for validation
coco_validation_annotation = "License-Plate-Object-Detection-6/valid/_annotations.coco.json"
coco_validation = CocoDetection(coco_validation_annotation)
coco_validation_path = "License-Plate-Object-Detection-6/valid/" #actual validation path

#Set Annotation path for testing
coco_testing_annotation = "License-Plate-Object-Detection-6/test/_annotations.coco.json"
coco_test = CocoDetection(coco_testing_annotation)
coco_test_path = "License-Plate-Object-Detection-6/test/" #actual test path

#Create the datasets (which need COCO conversions for Pytorch)
train_dataset = CustomCocoDetection(root=coco_train_path, annotationFile=coco_train_annotation, transform=transform, target_category_id=1)
validation_dataset = CustomCocoDetection(root=coco_validation_path, annotationFile=coco_validation_annotation, transform=transform, target_category_id=1)
test_dataset = CustomCocoDetection(root=coco_test_path, annotationFile=coco_testing_annotation, transform=transform, target_category_id=1)


#Create the dataloaders to start actually pulling data in

#Set an arbitrary batch size
batch_size = 8

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

#plt.show()


#Time to start constructing the actual network
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PlateIdentifier(nn.Module):
        def __init__(self, num_classes):
            super(PlateIdentifier, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #all of these images are 640x640, need to pad to maintain given a currnet convolve kernel of 16x16 pixels (3 ocrresponds to RGB)
            self.conv2 = nn.Conv2d(16, 31, 3, padding=1) #In channels have to correspond to previous layer out channel count
            self.pool = nn.MaxPool2d(2, 2) #2x2 pixel window that moves 2 pixels at a time
            
            
            #Upsample
            self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
            #Segmentation branch
            self.seg_conv = nn.Conv2d(31, num_classes, 1) #Segmentation head with channel adjustment for segmentation

            #Binary classification branch
            self.fc1 = nn.Linear(31 * 640 * 640, 1) #Simple 1 for binary classification
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x))) #Two relus, two pools
            
            x = self.upsample1(x)
            x = self.upsample2(x)
            
            #Segmentation branch output
            seg_output = self.seg_conv(x)
            #print("Segmentation Output Size:", seg_output.shape) #Shape right size?
            
            #Flatten for binary classification
            x = x.view(x.size(0), -1)
            binary_output = self.fc1(x) 
            
            return binary_output, seg_output
            #return x.squeeze(1), seg_output #Remove the single dimension, s and return the segmentation output as well

num_classes = 1
model = PlateIdentifier(num_classes)
    
#Define our loss function - using binary cross-entropy because plate is either there or it isn't
loss_fn = nn.BCEWithLogitsLoss() 
seg_loss_fn = nn.BCEWithLogitsLoss() #Segmentation loss


#Since we are identifying AND segmenting, get the combined loss
def combined_loss(binary_output, segmentation_output, binary_labels, segmentation_labels):
    binary_labels = binary_labels.unsqueeze(1)
    #Calculate BC loss
    binary_loss = loss_fn(binary_output, binary_labels.float())
    
    #Calculate seg loss
    segmentation_loss = seg_loss_fn(segmentation_output, segmentation_labels.float())
    
    #Combine losses 
    combined_loss = binary_loss + segmentation_loss
    
    return combined_loss

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if __name__ == "__main__":
    #Set up the tensorboard writer
    #Per epoch data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/plate_trainer_{}'.format(timestamp))
    epoch_number = 0
    epochs = 30
    best_vloss = 1_000_000

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        #Gradient tracking on, make a pass over the data!
        model.train(True)
        running_loss = 0.0
        
        #Also monitor accuracy
        #correct_predictions = 0
        #total_samples = 0

        for i, (inputs, labels, segmentation_labels) in enumerate(train_loader):
            #labels = labels.unsqueeze(1).float()
            #print(f'Batch {i+1}: Labels shape = {labels.shape}, Labels value = {labels}')
            optimizer.zero_grad()
            
            outputs, seg_outputs = model(inputs) #Binary inputs AND segmentation inputs
            
            binary_output = torch.sigmoid(outputs)
            
            #Verify shapes before calculating loss
            #print("Seg Output Size for Loss Calculation:", seg_outputs.shape) 
            #print("Seg Label Size for Loss Calculation:", segmentation_labels.shape)
           
            loss = combined_loss(binary_output, seg_outputs, labels, segmentation_labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            #Calculate the accuracy for each batch
            #binary_predictions = (binary_output > 0.5).float() #Convert probabilities to binary predictions with confidence threshold of 50%
            #correct_predictions += (binary_predictions == labels).sum().item() #Count correct predictions
            #total_samples += labels.size(0) #Total number of samples in the batch
            
            if i % 10 == 9: #Print every 10 mini batches
                avg_loss = running_loss / 10
                print(f'    batch {i+1} loss: {avg_loss}')
                writer.add_scalar('Loss/train', avg_loss, epoch_number * len(train_loader) + i + 1)
                running_loss = 0.0    
        
        #epoch_accuracy = correct_predictions / total_samples
        
        #Log the training accuracy for the epoch
        #writer.add_scalar("Accuracy/train", epoch_accuracy, epoch_number + 1)
        
        #Switch to evaluation mode
        model.eval()
        #Want to monitor accuracy as well during testing
        #vcorrect_predictions = 0
        #vtotal_samples = 0
        #Reduce memory consumption by disabling gradient computation
        with torch.no_grad():
            running_vloss = 0.0
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels, vseg_labels = vdata
                voutputs, vseg_outputs = model(vinputs)
                
                #Convert vlabels to a long tensor
                vlabels = vlabels.float()
                
                vloss = combined_loss(torch.sigmoid(voutputs), vseg_outputs, vlabels, vseg_labels)
                running_vloss += vloss.item() #Accumulate loss
                
                #Calculate the accuracy for each batch
                #vbinary_predictions = (voutputs > 0.5).float() #Convert probabilities to binary predictions with confidence threshold of 50%
                #vcorrect_predictions += (vbinary_predictions == vlabels).sum().item() #Count correct predictions
                #vtotal_samples += vlabels.size(0) #Total number of samples in the batch

        #vepoch_accuracy = vcorrect_predictions / vtotal_samples

        avg_vloss = running_vloss / len(validation_loader)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        
        #Log validation accuracy
        #writer.add_scalar("Accuracy/validation", vepoch_accuracy, epoch_number + 1)

        #Log running loss averaged per batch for both the training AND validation
        writer.add_scalars('Training vs. Validation Loss',
                           { 'Training' : avg_loss, 'Validation' : avg_vloss },
                           epoch_number + 1)
        writer.flush()

        #Track best performance and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}.pth'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
            
        #print(f'Training Accuracy: {epoch_accuracy*100:.2f}%, Validation Accuracy: {vepoch_accuracy*100:.2f}%')
        epoch_number += 1
        
        #Now we test to see how well the model performs
        model.eval() #Should stay the same? Can't hurt to be explicit
        #tcorrect_predictions = 0
        #ttotal_samples = 0
        tp_count = 0
        fp_count = 0
        
        for test_inputs, test_labels, test_seg_labels in test_loader:
            test_outputs, test_seg_outputs = model(test_inputs)
            test_binary_predictions = (test_outputs > 0.5).float()
            #tcorrect_predictions += (test_binary_predictions == test_labels).sum().item()
            #ttotal_samples += test_labels.size(0)
            
            #Calculate true and false positives
            tp_count += ((test_binary_predictions == test_labels) & (test_labels == 1)).sum().item()
            fp_count += ((test_binary_predictions != test_labels) & (test_labels == 0)).sum().item()
            
            #Calculate precision
            precision = tp_count / (tp_count + fp_count) if(tp_count + fp_count) > 0 else 0.0 #Prevent divide by 0 issues
            
            print(f'Precision: {precision * 100:.2f}%')
            
            #Display some images from testing
            num_display = 3
            for i in range(num_display):
                image = test_inputs[i].permute(1, 2, 0).numpy() #Convert tensor to numpy array
                plt.subplot(2, num_display, i+1)
                plt.imshow(image)
                plt.axis('off')
                plt.title('Image')
                
                mask = test_seg_outputs[i].detach().permute(1, 2, 0).numpy() #Convert seg mask to numpy array
                plt.subplot(2, num_display, i + 1 + num_display)
                plt.imshow(mask.squeeze(), cmap='gray')
                plt.axis('off')
                plt.title('Segmentation Mask')
            plt.show()
            break
            
        #test_accuracy = tcorrect_predictions / ttotal_samples
        #print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
            
          
