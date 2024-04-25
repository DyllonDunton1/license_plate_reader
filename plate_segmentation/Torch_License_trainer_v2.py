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
project = rf.workspace("cj-santos-e0nrn").project("license-plate-detection-3lgox")
version = project.version(3)
dataset = version.download("coco")

#Since it is annotated in COCO, need to pull in additionall libraries

try:
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    
except:
    print("Unable to import necessary libraries")

#Since the data we are using is annotated beyond what we want, need to pull out the proper annotations

#Define some transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

#Set the dataset directories
#Annotation path for training
coco_train_annotation = "License-Plate-Object-Detection-6/train/_annotations.coco.json"
coco_train_path = "License-Plate-Object-Detection-6/train/" #actual image path

#Annotation path for validation
coco_validation_annotation = "License-Plate-Object-Detection-6/valid/_annotations.coco.json"
coco_validation_path = "License-Plate-Object-Detection-6/valid/" #actual validation path

#Annotation path for testing
coco_testing_annotation = "License-Plate-Object-Detection-6/test/_annotations.coco.json"
coco_test_path = "License-Plate-Object-Detection-6/test/" #actual test path

#Load the dataset
train_dataset = torchvision.datasets.CocoDetection(root=coco_train_path, annFile=coco_train_annotation, transform=transform)
val_dataset = torchvision.datasets.CocoDetection(root=coco_validation_path, annFile=coco_validation_annotation, transform=transform)
test_dataset = torchvision.datasets.CocoDetection(root=coco_test_path, annFile=coco_testing_annotation, transform=transform)

#Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  #4 outputs for bounding box coordinates (x1, y1, x2, y2)
        )

        #Classification head for license plate and numbers
        self.cls_head = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),  #Number of classes (license plate + numbers)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)  #Flatten the feature map for dense layers

        #Bounding box regression predictions
        bbox_preds = self.reg_head(x)

        #Class predictions (license plate and numbers)
        cls_preds = self.cls_head(x)

        return bbox_preds, cls_preds
        
#Define training parameters
num_classes = 2
model = LicensePlateDetector(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#Try to get annotation keyword directly
import json

with open(coco_train_annotation, 'r') as f:
    annotation_data = json.load(f)

#Access the training list directly
annotations = annotation_data["annotations"]

#Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        
        #Send to GPU just in case they aren't already there
        images = images.to(device)
        
        bbox_preds, cls_preds = model(images)
        bbox_targets = []
        cls_targets = []

        #Filter annotations based on the batch size
        batch_start = batch_idx * train_loader.batch_size
        batch_end = batch_start + len(targets)
        batch_annotations = annotations[batch_start:batch_end]
        
        for annotation in annotations:
            bbox = annotation["bbox"]
            bbox_targets.append(bbox) 
            cls_label = annotation["category_id"]  
            cls_targets.append(cls_label) 
            
        #Convert bbox_targets and cls_targets to tensors
        bbox_targets = torch.tensor(bbox_targets, dtype=torch.float32, device=device)
        cls_targets = torch.tensor(cls_targets, dtype=torch.long, device=device)
        
        #Calculate bounding box regression loss 
        #bbox_targets are ground truth bounding box coordinates
        
        #Check dimensions
        print("bbox_preds shape:", bbox_preds.shape)
        print("bbox_targets shape:", bbox_targets.shape)
        
        bbox_loss = F.mse_loss(bbox_preds, bbox_targets)
        
        #Calculate classification loss 
        #cls_targets are ground truth class labels
        cls_loss = criterion(cls_preds, cls_targets)
        
        total_loss = bbox_loss + cls_loss
        total_loss.backward()
        optimizer.step()

    #Validation after each epoch
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            bbox_preds, cls_preds = model(val_images)
            bbox_loss = F.smooth_l1_loss(bbox_preds, bbox_targets)
            cls_loss = criterion(cls_preds, cls_targets)
            val_loss = bbox_loss + cls_loss
            val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {torch.mean(torch.tensor(val_losses)):.4f}")
    
#After training and eval, visualize a test prediction
 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

model.eval()

def draw_boxes(image, boxes, labels):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))  #Convert from tensor format to numpy format (HWC)
    for box, label in zip(boxes, labels):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f'{label}', color='r', verticalalignment='bottom')
    plt.show()

#Iterate through the test dataset and visualize predictions
for images, targets in test_loader:
    image = images[0]  #Assuming batch size is 1
    image_tensor = image.unsqueeze(0).to(device)  #Add batch dimension and move to device if using GPU
    bbox_preds, cls_preds = model(image_tensor)
    
    #Process predictions to get bounding box coordinates and class labels
    #Assuming bbox_preds is in (x1, y1, x2, y2) format
    boxes = bbox_preds.squeeze(0).detach().cpu().numpy()
    labels = cls_preds.squeeze(0).argmax(dim=1).detach().cpu().numpy()  #Assuming class predictions are in softmax format
    
    #Draw bounding boxes on the image
    draw_boxes(image, boxes, labels)
