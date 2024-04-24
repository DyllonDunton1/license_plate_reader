import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from Torch_License_Trainer import PlateIdentifier

#Load trained model
model = PlateIdentifier(1)
model.load_state_dict(torch.load('model_20240424_022905_29.pth'))
model.eval()

#Preprocess image
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
    ])
    
image_path = 'License-Plate-Object-Detection-6/test/42ap97bwjxea1_jpg.rf.ce9f0930d628a471bde2c9fbc4486ea2.jpg'
input_image = Image.open(image_path)
input_tensor = transform(input_image).unsqueeze(0)

#Infer
with torch.no_grad():
    output_mask = model(input_tensor)
    
#Post process segmentation mask
threshold = 0
binary_mask = (output_mask > threshold).float()

#Extract license plate region
plate_image = input_image.copy()
plate_image.putalpha(255 * binary_mask.squeeze().numpy().astype(np.uint8)) #Apply mask to plate

#Save extracted image!

plate_image.save('test.png')
