##This model trains a Pytorch license plate image segmentation model

##Try to import Pytorch and check CUDA availability

try:
    import torch
    import torchvision
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
project = rf.workspace("cj-santos-e0nrn").project("license-plate-detection-3lgox")
version = project.version(3)
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
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    
except:
    print("Unable to import necessary libraries")

#Since the data we are using is annotated beyond what we want, need to pull out the proper annotations
class CustomCocoDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Extract target information from annotations
        bbox_list = []
        cls_list = []
        for ann in anns:
            bbox = ann['bbox']  # Extract bounding box coordinates
            cls_label = ann['category_id']  # Extract class label
            bbox_list.append(bbox)
            cls_list.append(cls_label)

        img_path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Convert targets to tensors
        bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
        cls_tensor = torch.tensor(cls_list, dtype=torch.long)

        return img, img_id, (bbox_tensor, cls_tensor)

    def __len__(self):
        return len(self.ids)

#Define some transforms
transform = transforms.Compose([
    #transforms.Resize((320, 320)),
    transforms.ToTensor()
])

#Set the dataset directories
#Annotation path for training
coco_train_annotation = "License-Plate-Detection-3/train/_annotations.coco.json"
coco_train = COCO(coco_train_annotation)
coco_train_path = "License-Plate-Detection-3/train/"  #actual image path

#Annotation path for validation
coco_validation_annotation = "License-Plate-Detection-3/valid/_annotations.coco.json"
coco_validation = COCO(coco_validation_annotation)
coco_validation_path = "License-Plate-Detection-3/valid/"  #actual validation path

#Annotation path for testing
coco_testing_annotation = "License-Plate-Detection-3/test/_annotations.coco.json"
coco_test = COCO(coco_testing_annotation)
coco_test_path = "License-Plate-Detection-3/test/"  #actual test path

#Create the datasets using the custom dataset class
train_dataset = CustomCocoDataset(root=coco_train_path, annFile=coco_train_annotation, transform=transform)
val_dataset = CustomCocoDataset(root=coco_validation_path, annFile=coco_validation_annotation, transform=transform)
test_dataset = CustomCocoDataset(root=coco_test_path, annFile=coco_testing_annotation, transform=transform)

#Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)


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

#Define Model parameters
class LicensePlateDetector(nn.Module):
    def __init__(self, num_classes):
        super(LicensePlateDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        #Bounding box regression head
        self.reg_head = nn.Sequential(
            nn.Linear(256 * 40 * 40, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  #4 outputs for bounding box coordinates (x1, y1, x2, y2)
        )

        '''
        #Classification head for license plate and numbers
        self.cls_head = nn.Sequential(
            nn.Linear(256 * 40 * 40, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),  #Number of classes (license plate + numbers)
        )
        '''
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print("Shape after conv3 and pool:", x.shape)
        x = x.view(-1, 256 * 40 * 40)  #Flatten the feature map for dense layers
        #print("Flattened shape:", x.shape)

        #Bounding box regression predictions
        bbox_preds = self.reg_head(x)

        #Class predictions (license plate and numbers)
        #cls_preds = self.cls_head(x)

        return bbox_preds#, cls_preds
        
#Define training parameters
num_classes = 1
model = LicensePlateDetector(num_classes)
device = torch.device('cuda')
model.to(device)

#Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.00000001)
#criterion = nn.CrossEntropyLoss()

#Try to get annotation keyword directly
import json

with open(coco_train_annotation, 'r') as f:
    annotation_data = json.load(f)

#Access the training list directly
annotations = annotation_data["annotations"]

best_vloss = 10000

if __name__ == "__main__":
    #Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print("EPOCH:", epoch + 1)
        model.train()
        for images, img_id, targets in train_loader:
            optimizer.zero_grad()
            #print("Image size", images.size(0))
            #print("Image shape", images.shape)
            #print("target shape", len(targets))
            #print(targets)
            #print("Iteration number:", batch_idx)
            #targets = torch.tensor(targets)
            #print("Target after tensor:", targets.shape)
            #Send to GPU just in case they aren't already there
            images = images.to(device)
            
            bbox_preds = model(images)
            bbox_targets = []
            #cls_targets = []
            
            #print("BBOX PREDS SHAPE:", bbox_preds.shape)
            #print("CLS PREDS SHAPE:", cls_preds.shape)

            bbox_targets = targets[0][0]
           
            #print("BBox target shape: , Image ID: ", bbox_targets.shape[0], img_id)
            #if bbox_targets.shape[0] != 1:
            #     continue
            
            bbox_targets = bbox_targets.to(device)
            #cls_targets = targets[1][0].to(device)
            #cls_preds = cls_preds[0]
            
            #print("Split bbox_targets", bbox_targets)
            #print("Split cls_targets", cls_targets)
            #print("Class predictions", cls_preds)
            
            #Convert bbox_targets and cls_targets to tensors
            #bbox_targets = torch.tensor(bbox_targets, dtype=torch.float32, device=device)
            #cls_targets = torch.tensor(cls_targets, dtype=torch.float, device=device)
            #cls_preds = torch.tensor(cls_preds, dtype=torch.float, device=device)
            
            #Calculate bounding box regression loss 
            #bbox_targets are ground truth bounding box coordinates
            
            #Check dimensions
            #print("bbox_preds shape:", bbox_preds.shape)
            #print("bbox_targets shape:", bbox_targets.shape)
            #print("cls_preds shape:", cls_preds.shape)
            #print("cls_targets shape:", cls_targets.shape)
            
            bbox_loss = F.mse_loss(bbox_preds, bbox_targets)
            
            #Calculate classification loss 
            #cls_targets are ground truth class labels
            #cls_loss = criterion(cls_preds, cls_targets)
            
            total_loss = bbox_loss# + cls_loss
            
            total_loss.backward()
            optimizer.step()
        
        print("Epoch: training loss:", (epoch+1), total_loss.item())
        #Validation after each epoch
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vimages, vimg_id, vtargets in val_loader:
                vimages = vimages.to(device)
            
                vbbox_preds = model(vimages)
                vbbox_targets = []
                #cls_targets = []
                
                #print("BBOX PREDS SHAPE:", bbox_preds.shape)
                #print("CLS PREDS SHAPE:", cls_preds.shape)

                vbbox_targets = vtargets[0][0]
               
                #print("BBox target shape: , Image ID: ", vbbox_targets.shape[0], vimg_id)
                #if bbox_targets.shape[0] != 1:
                #     continue
                
                vbbox_targets = vbbox_targets.to(device)
                #cls_targets = targets[1][0].to(device)
                
                #print("Split bbox_targets", vbbox_targets)
                #print("Split cls_targets", cls_targets)
                #Convert bbox_targets and cls_targets to tensors
                #bbox_targets = torch.tensor(bbox_targets, dtype=torch.float32, device=device)
                #cls_targets = torch.tensor(cls_targets, dtype=torch.long, device=device)
                
                #Calculate bounding box regression loss 
                #bbox_targets are ground truth bounding box coordinates
                
                #Check dimensions
                #print("bbox_preds shape:", bbox_preds.shape)
                #print("bbox_targets shape:", bbox_targets.shape)
                
                vbbox_loss = F.mse_loss(vbbox_preds, vbbox_targets)
                
                #print("Validation loss:", vbbox_loss.item())
                val_losses.append(vbbox_loss.item())
        
        #print("Validation loss:", val_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {torch.mean(torch.tensor(val_losses)):.4f}")
        
        #Track best performance and save the model's state
        if vbbox_loss.item()< best_vloss:
            best_vloss = vbbox_loss.item()
            model_path = 'model_{}.pth'.format(epoch)
            torch.save(model.state_dict(), model_path)
        
    #After training and eval, visualize a test prediction
     
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    model.eval()

    def draw_boxes(image, boxes):
        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0))  #Convert from tensor format to numpy format (HWC)
        for box in boxes:
            x, y, w, h = box
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

    #Iterate through the test dataset and visualize predictions
    for timages, timg_id, ttargets in test_loader:
        timage = timages[0]  #Assuming batch size is 1
        timage_tensor = timage.to(device)  #Add batch dimension and move to device if using GPU
        tbbox_preds = model(timage_tensor)
        
        #Process predictions to get bounding box coordinates and class labels
        #Assuming bbox_preds is in (x1, y1, x2, y2) format
        tboxes = tbbox_preds.detach().cpu().numpy()
        #tlabels = tcls_preds.squeeze(0).argmax(dim=1).detach().cpu().numpy()  #Assuming class predictions are in softmax format
        
        #Draw bounding boxes on the image
        draw_boxes(timage, tboxes)