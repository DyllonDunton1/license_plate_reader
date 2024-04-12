#%%
import csv
import numpy as np
import matplotlib.pyplot as plt
import microtorch as t
import torch


mapping = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e', 40:'f', 41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'}

test_labels = []
test_imgs = []
train_labels = []
train_imgs = []

def csv_read(path):
    labels = []
    imgs = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            label = int(row[0])
            if label < 36:
                labels.append(label)
                ints = [int(pixel) for pixel in row]
                imgs.append((np.asarray(ints[1:]).reshape(28,28)).T)
    return  labels, imgs          

def convert_labels_to_one_hot_vectors(input):
    return np.eye(36)[input]

test_labels_indeces, test_imgs = csv_read('/home/dunto/ece590/emnist-balanced-test.csv')
train_labels_indeces, train_imgs = csv_read('/home/dunto/ece590/emnist-balanced-train.csv')

test_labels = convert_labels_to_one_hot_vectors(test_labels_indeces)
train_labels = convert_labels_to_one_hot_vectors(train_labels_indeces)

test_len = len(test_labels_indeces)
train_len = len(train_labels_indeces)
assert test_len == len(test_imgs)
assert train_len == len(train_imgs)
print(f'Test images: {len(test_labels)}\nTrain images: {len(train_labels)}')
print(f'Train to Test Ration = {train_len/test_len}:1.0')
print("Done Loading Images into test and train")

#%%

index = 80

## Visualizing matrices
fig, ax = plt.subplots()
ax.axis('off')
print(f'Label: {mapping[train_labels_indeces[index]]}')
ax.imshow(train_imgs[index], cmap='gray')

# %%
from scipy import signal
#Test scipy convolve
kernel = np.ones((3,3))
out = signal.convolve2d(train_imgs[index], kernel, boundary='fill', mode='same')

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(out, cmap='gray')
print(out.shape)
# %%


from microtorch_nn import Convolution
from microtorch_nn import ReLU
from microtorch_nn import MaxPool
from microtorch_nn import Flatten
from microtorch_nn import Linear
from microtorch_nn import SoftMax
from microtorch_nn import Sequential
from microtorch_nn import Argmax

first_convolution = Convolution(28, 7) #28,28
first_activation = ReLU() #28,28
first_maxpool = MaxPool() #14,14

second_convolution = Convolution(14, 7) #14,14
second_activation = ReLU() #14,14
second_maxpool = MaxPool() #7,7

final_flatten = Flatten() #49,1
final_linear_extension = Linear(49, 36) #36,1
final_softmax = SoftMax() #36,1


model = Sequential(first_convolution, 
                    first_activation,
                    first_maxpool,
                    second_convolution,
                    second_activation,
                    second_maxpool,
                    final_flatten,
                    final_linear_extension,
                    final_softmax)
'''
prediction = model(test_imgs[index])

print(prediction)
print(prediction.shape)

predicted_label = mapping[np.argmax(prediction.value)]

print(f"This img is of a '{predicted_label}'")
'''
# %%

from microtorch import Tensor

def loss(predicted_labels, true_labels):
    y = np.asarray(true_labels)
    yhat = predicted_labels
    print(yhat.value)
    print(f'y: {y.shape} | yhat: {yhat.shape}')
    assert y.shape == yhat.shape
    return -(y*yhat).sum()

def train_by_gradient_descent(model, loss, train_imgs, train_labels, lr=1e-4):
    #Get the initial prediction to train from
    prediction = model(train_imgs)
    #Calculate initial loss
    loss_t = loss(prediction, train_labels)
    loss_t.backward(1)
    loss_t_minus_1 = 2*loss_t.value
    while np.abs(loss_t.value - loss_t_minus_1)/loss_t.value > 1e-2:
        for param in model.parameters():
            assert param.grad is not None
            #print("before:", id(param))
            param.value = param.value - lr * param.grad  # Gradient descent
            #print("after:", id(param))
        loss_t.zero_grad()
        # Recompute the gradients
        predicted_labels = model(train_imgs)
        loss_t_minus_1 = loss_t.value
        loss_t = loss(predicted_labels, train_labels)
        loss_t.backward(1) # Compute gradients for next iteration
        
        # If loss increased, decrease lr. Works for gradient descent, not for stochatic gradient descent.
        if loss_t.value > loss_t_minus_1:
            lr = lr / 2
        
        ### DEBUGing information
        iswrong = (train_labels * predicted_labels.value.ravel()) < 0
        misclassified = (iswrong).sum() / iswrong.shape[0]
        print(f"loss: {loss_t.value:04.04f}, delta loss: {loss_t.value - loss_t_minus_1:04.04f}," 
              f"train misclassified: {misclassified:04.04f}")
        
            
    return model

trained_model = train_by_gradient_descent(model, loss, train_imgs, train_labels)