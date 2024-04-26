import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from Torch_License_Trainer_v3 import LicensePlateDetector
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Load trained model
model = LicensePlateDetector(1)
model.load_state_dict(torch.load('model_1.pth'))
model.eval()

#Preprocess image
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
    ])
    
image_path = '/mnt/c/Users/woody/Downloads/IMG_0539.jpg'
input_image = Image.open(image_path)
input_tensor = transform(input_image)

def draw_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))  #Convert from tensor format to numpy format (HWC)
    for box in boxes:
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig('example.png', bbox_inches='tight') #Save the test image
    plt.show()

tbbox_preds = model(input_tensor)

#Process predictions to get bounding box coordinates and class labels
#Assuming bbox_preds is in (x1, y1, x2, y2) format
tboxes = tbbox_preds.detach().cpu().numpy()
#tlabels = tcls_preds.squeeze(0).argmax(dim=1).detach().cpu().numpy()  #Assuming class predictions are in softmax format

#Draw bounding boxes on the image
draw_boxes(input_tensor, tboxes)
