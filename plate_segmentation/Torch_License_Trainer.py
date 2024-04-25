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
    from pycocotools.coco import COCO 
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
        self.coco = COCO(annotationFile)
        self.transform = transform
        self.target_category_id = target_category_id
        self.image_size = image_size
        self.filtered_annotations = self.filter_annotations()
        
    def filter_annotations(self):
        filtered_annotations = []
        for ann in self.coco.dataset['annotations']:
            if ann['category_id'] == self.target_category_id: #Category ID 1 corresponds to the license plate mask
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
            seg_points = np.array(annotation['segmentation'][0]).reshape((-1 ,2)) #Reshape to x,y for fillPoly (2Darray w/ 2 columns)
            seg_points = np.round(seg_points).astype(np.int32)
            cv.fillPoly(mask, [seg_points], 1) #Fill license plate region with 1
        
        if self.transform is not None:
            img = self.transform(img)
            
        label = annotation['label']
        segmentation_label = mask
        return img, label, segmentation_label
    
    def __len__(self):
        return len(self.filtered_annotations)


transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()])

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

#Create the datasets (which need COCO conversions for Pytorch)
train_dataset = CustomCocoDetection(root=coco_train_path, annotationFile=coco_train_annotation, transform=transform, target_category_id=1)
validation_dataset = CustomCocoDetection(root=coco_validation_path, annotationFile=coco_validation_annotation, transform=transform, target_category_id=1)
test_dataset = CustomCocoDetection(root=coco_test_path, annotationFile=coco_testing_annotation, transform=transform, target_category_id=1)


#Create the dataloaders to start actually pulling data in

#Set an arbitrary batch size
batch_size = 2

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
        def __init__(self):
            super(PlateIdentifier, self).__init__()
            
            #Binary classification layer
            self.fc = nn.Linear(640 * 640 * 3, 1) #640x640 input image, 3 channel RGB
            
            #Segmentation layers
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 31, 3, padding=1)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.seg_conv = nn.Conv2d(31, 2, 1) #2 segmentation classes, the license plate and numbers

        def forward(self, x):
            #Binary classification branch
            x_binary = x.view(x.size(0), -1) #Flatten the input for fully connected layer
            x_binary = torch.sigmoid(self.fc(x_binary)) #Sigmoid for binary classification
            
            #Segmentation Branch
            x_seg = F.relu(self.conv1(x))
            x_seg = F.relu(self.conv2(x_seg))
            x_seg = self.upsample(x_seg)
            #x_seg = F.interpolate(x_seg, size=(640, 640), mode='bilinear', align_corners=True)  #Adjust spatial dimensions
            x_seg = self.seg_conv(x_seg)
            x_seg = F.interpolate(x_seg, size=(640, 640), mode='bilinear', align_corners=True)  #Resize to match target size
            return x_binary, x_seg

model = PlateIdentifier()
    
#Define our loss functions
binary_loss_fn = nn.BCEWithLogitsLoss() #Binary crossentropy for binary classification
seg_loss_fn = nn.CrossEntropyLoss() #Multiple class crossentropy

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def calculate_recall(true_positives, false_negatives):
    if true_positives + false_negatives == 0:
        return 0.0  #To handle cases where the denominator is zero
    recall = true_positives / (true_positives + false_negatives)
    return recall


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

        for i, (inputs, labels, segmentation_labels) in enumerate(train_loader):
            #print(f'Batch {i+1}: Labels shape = {labels.shape}, Labels value = {labels}')
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            
            binary_output, seg_outputs = model(inputs) #Get binary and segmentation info
            
            #Calculate binary classification loss
            binary_loss = binary_loss_fn(binary_output, labels.float())
            
            #Calculate segmentation loss
            seg_loss = seg_loss_fn(seg_outputs, segmentation_labels.long())
   
           
            total_loss = binary_loss + seg_loss
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            
            if i % 10 == 9: #Print every 10 mini batches
                avg_loss = running_loss / 10
                print(f'    batch {i+1} loss: {avg_loss}')
                writer.add_scalar('Loss/train', avg_loss, epoch_number * len(train_loader) + i + 1)
                running_loss = 0.0    
        
        #Switch to evaluation mode
        model.eval()
        #Want to monitor accuracy as well during testing
        #vcorrect_predictions = 0
        #vtotal_samples = 0
        
        #Monitor recall
        true_positives = 0
        false_positives = 0
       
        #Reduce memory consumption by disabling gradient computation
        with torch.no_grad():
            running_vloss = 0.0
            for i, (vinputs, vlabels, vseg_labels) in enumerate(validation_loader):
                vlabels = vlabels.unsqueeze(1)
                vbinary_output, vseg_outputs = model(vinputs)
                
                vbinary_predictions = (vbinary_output > 0.5).float() #Convert probabilities to binary
                
                #Calculate binary classification loss
                vbinary_loss = binary_loss_fn(vbinary_output, vlabels.float())
                
                #Calculate segmentation loss
                vseg_loss = seg_loss_fn(vseg_outputs, vseg_labels.long())
       
                vlabels = vlabels.squeeze(1)
                #Update true positive and false positive counts
                true_positives = ((vbinary_predictions == 1) & (vlabels == 1)).sum().item()
                false_positives = ((vbinary_predictions == 0) & (vlabels == 1)).sum().item()
               
                vtotal_loss = vbinary_loss + vseg_loss
                
                optimizer.step()
                
                running_vloss += vtotal_loss.item() #Accumulate loss
                
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
            
        recall = calculate_recall(true_positives, false_positives)
        print("Recall:", recall)
            
        epoch_number += 1
        

        #Now we test to see how well the model performs
        
        #Define colors for different classes or segments
        class_colors = {
            0: [0, 0, 0],        # Background (black)
            1: [255, 0, 0],      # License plate (red)
            2: [0, 255, 0],      # Numbers (green)
        }
                
        #Convert segmentation mask to RGB format based on class colors
        def mask_to_rgb(mask):
            rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for class_id, color in class_colors.items():
                rgb_mask[mask == class_id] = color
            return rgb_mask
        
        
        model.eval() #Should stay the same? Can't hurt to be explicit
        
        with torch.no_grad():
            for test_inputs, test_labels, test_seg_labels in test_loader:
                test_outputs, test_seg_outputs = model(test_inputs)
                test_binary_predictions = (test_outputs > 0.5).float()

                
                #Display some images from testing
                num_display = min(3, len(test_inputs))
                for i in range(num_display):
                    image = test_inputs[i].permute(1, 2, 0).numpy() #Convert tensor to numpy array
                    plt.subplot(2, num_display, i+1)
                    plt.imshow(image)
                    plt.axis('off')
                    plt.title('Image')
                    
                    mask = test_seg_outputs[i, 1].detach().numpy() #Convert seg mask to numpy array
                    rgb_mask = mask_to_rgb(mask)
                    plt.subplot(2, num_display, i + 1 + num_display)
                    plt.imshow(rgb_mask)
                    plt.axis('off')
                    plt.title('Segmentation Mask')
                plt.show()
                break

          
