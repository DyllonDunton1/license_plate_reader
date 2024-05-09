#%%
import csv
import numpy as np
import matplotlib.pyplot as plt
import microtorch as t


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

test_labels_indeces, test_imgs = csv_read('emnist-balanced-test.csv')
train_labels_indeces, train_imgs = csv_read('emnist-balanced-train.csv')

test_labels = convert_labels_to_one_hot_vectors(test_labels_indeces)
train_labels = convert_labels_to_one_hot_vectors(train_labels_indeces)

test_len = len(test_labels_indeces)
train_len = len(train_labels_indeces)
assert test_len == len(test_imgs)
assert train_len == len(train_imgs)
print(f'Test images: {len(test_labels)}\nTrain images: {len(train_labels)}')
print(f'Train to Test Ration = {train_len/test_len}:1.0')
#print(train_labels)

mean_image = np.array(train_imgs).mean()
std_image = np.array(train_imgs).std()
#zeros = np.zeros(std_image.shape)
#std_image = np.where(zeros == std_image, 1, std_image)
print(mean_image.shape, std_image.shape)

train_imgs = (np.array(train_imgs) - mean_image)/std_image
print(train_imgs.shape)

print("Done Loading Images into test and train")

#%%
'''
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
'''
# %%

from microtorch_nn import Convolution
from microtorch_nn import ReLU
from microtorch_nn import MaxPool
from microtorch_nn import Flatten
from microtorch_nn import Linear
from microtorch_nn import SoftMax
from microtorch_nn import Sequential

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
                   first_convolution,
                   first_convolution,
                    first_activation,
                    first_maxpool,
                    second_convolution,
                    second_convolution,
                    second_convolution,
                    second_convolution,
                    second_activation,
                    second_maxpool,
                    final_flatten,
                    final_linear_extension,
                    final_softmax)

#prediction = model(test_imgs[index])

#print(prediction)
#print(prediction.shape)

#predicted_label = mapping[np.argmax(prediction.value)]

#print(f"This img is of a '{predicted_label}'")

# %%

from microtorch import Tensor

def loss(predicted_labels, true_labels):
    y = true_labels
    yhat = predicted_labels
    #breakpoint()
    #print(f'y: {y.shape} | yhat: {yhat.shape}')
    assert y.shape == yhat.shape
    out = -(y*(t.log(yhat))).sum()
    #print("Done loss")
    return out

iterations = []
loss_array = []
def train_by_gradient_descent(model, loss, train_imgs, train_labels, lr=0.001):
    #Get the initial prediction to train from
    print("\n\n\n\n----------------------------------\nBegin Training using Stochastic Gradient Descent!\nWe go through the dataset every epoch...\n-----------------")
    shuffle = np.arange(len(train_imgs))
    np.random.shuffle(shuffle)
    #print(shuffle)
    #print(len(shuffle))
    #print(shuffle.shape)
    prediction = model(t.Tensor(train_imgs[int(shuffle[-1])]))
    #print("Done prediction")
    #Calculate initial loss
    loss_t = loss(prediction, train_labels[int(shuffle[-1])])
    print("Initial Loss: " + str(loss_t.value))
    loss_t.backward(1)
    loss_t_minus_1 = 2*loss_t.value
    loss_last_iter = loss_t.value
    epoch = 0
    #breakpoint()
    iterations.append(epoch)
    loss_array.append(loss_t.value)
    
    while abs(loss_t.value - loss_t_minus_1)> 1e-3:
        #LOOP HERE, BUT SHUFFLE SO THAT THEY ARE RANDOM
        epoch += 1
        if epoch != 1:
            loss_t_minus_1 = mean_loss_on_the_dataset
        print(f"Current Epoch: {epoch}")
        print(f"Learning Rate: {lr}")
        total_loss_on_the_dataset = loss_t.value
        for iter, index in enumerate(shuffle[:-1]):
            if iter%int(train_len/10) == 0:
                print(f"{iter} out of {train_len} | current loss: {(total_loss_on_the_dataset/(iter+1)):04.04f}...")
            #print("Params: " + str(list(model.named_parameters())))
            for param in model.parameters():
                assert param.grad is not None
                #print("before:", id(param))
                param.value = param.value - lr * np.minimum(np.maximum(param.grad, -1), 1)  # Gradient descent # gradient clipping
                #print("after:", id(param))
            loss_t.zero_grad()
            # Recompute the gradients
            predicted_labels = model(t.Tensor(train_imgs[int(index)]))
            
            loss_t = loss(predicted_labels, train_labels[int(index)])
            total_loss_on_the_dataset += loss_t.value
            loss_t.backward(1) # Compute gradients for next iteration

            ### DEBUGing information
            #print(predicted_labels.value.ravel())
            #iswrong = (train_labels.ravel() * predicted_labels.value.ravel()) < 0
            #misclassified = (iswrong).sum() / iswrong.shape[0]
            
        #breakpoint()
        mean_loss_on_the_dataset  = total_loss_on_the_dataset / (len(shuffle))
        print("--------------------------------------------------------")
        print(f"loss: {mean_loss_on_the_dataset:04.04f}, delta loss: {mean_loss_on_the_dataset - loss_t_minus_1:04.04f}")
        print(f'Current Epoch: {epoch} | Iteration: {iter}')
        iterations.append(epoch)
        loss_array.append(mean_loss_on_the_dataset)
        if loss_t.value > loss_t_minus_1:
                lr = lr / 2

        # If loss increased, decrease lr. Works for gradient descent, not for stochatic gradient descent.
        
    return model

trained_model = train_by_gradient_descent(model, loss, train_imgs, train_labels)



#%% VALIDATION

print("--------------------------------------------------------------------------\nVALIDATING...")

correct = 0
for img, lab in zip(test_imgs, test_labels_indeces):
    pred = trained_model(t.Tensor(img))
    pred_val = np.argmax(pred.value)
    #print(pred)
    print(pred_val, lab)
    if pred_val == lab:
        correct += 1 
print(f'Correct: {correct}')
accuracy = correct/test_len
print(f'Accuracy: {(accuracy*100):04.02f}%')

l = np.array(loss_array)
t = np.array(iterations)

plt.plot(t,l)
plt.title("Loss Over Time")
#plt.ylim(np.min(l), np.max(l))
plt.show()

#%% EXPORT MODEL
inputs = ""
while(True):
    inputs  = input("\n\n\nDo you want to save this model? (Y/n): ")
    if (inputs in ("Y","n")):
        break
    else:
        print("Not an answer...")
if inputs == "Y":
    save_name = input("\n What do you want to name the file as?: ")
    print("Saving Model...")

print("Have a nice day!")
