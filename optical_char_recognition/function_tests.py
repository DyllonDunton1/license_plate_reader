'''
import numpy as np
import matplotlib.pyplot as plt
import microtorch as t

#convolve testing (Asize - wsize + 1)
#forward
A = np.array([[1,2,3,4],
             [4,5,6,7],
             [7,8,9,10],
             [10,11,12,13]])
w = np.array([[1, 2,3],
             [3,4,5],
             [5,6,7]])


wsize does not change, and output must be original a size, then:

    Apadsize - wsize + 1 = Asize
    x - 3 + 1 = 4
    x = 6


    x - 7 + 1 = 28
    x = 34


f = t.conv_for(A, w)
dldA, dldk = t.conv_vjp(1, A, w)

print(f)

yhat = [.2, .1, .45, .25]
y = [1, 0, 0, 0]

out = -(y*(t.log(yhat))).sum()
print(out)


print(t.softmax(t.Tensor(np.array([-0.1, 3.8, 1.1, -0.3]))))

print(t.flat_for(np.array([[1,2,3,4,5,6,7],
                                   [1,2,3,4,5,6,7],
                                   [1,2,3,4,5,6,7],
                                   [1,2,3,4,5,6,7],
                                   [1,2,3,4,5,6,7],
                                   [1,2,3,4,5,6,7],
                                   [1,2,3,4,5,6,7],])))

print(t.absmax(np.array([.1, .2, .9, .3, .01, .89999999999])))


print(t.maxpool_for(np.array([[1,2,3,4,5,6,7,8],
                            [1,2,3,4,5,6,7,8],
                            [1,2,3,4,5,6,7,8],
                            [1,2,3,4,5,6,7,8],
                            [1,2,3,4,40,6,7,8],
                            [1,2,3,4,5,6,7,8],
                            [1,2,3,4,5,6,7,8],
                            [1,2,3,4,5,6,7,8]])))



def maxpool_vjp(dldf, x):
    #print("maxpool vjp")
    dldf_upsampled = np.kron(dldf, np.ones((2,2)))
    dxdl = np.where(dldf_upsampled == x, dldf_upsampled, 0)
    
    return dxdl,



dldf = np.array([[1,2],
                 [3,4]])

x = np.array([[3,1,1,3],
                [1,1,1,1],
                [1,1,1,1],
                [3,1,1,3]])



dldx = t.maxpool_vjp(dldf, x)[0]
print(dldx)



dldf = np.array([[5,10,15,20]])
x = np.array([[1,2],
                 [3,4]])

dldx = t.flat_vjp(dldf, x)
print(dldx)


def amax_vjp(dldf, x):
    #breakpoint()
    dldf_extend = np.ones(x.shape)*dldf
    dldx = np.where(x == dldf_extend, dldf_extend, 0)
    #breakpoint()
    return dldx,


x = np.array([1,2,15,5])
dldf = 10.1




dldx = t.amax_vjp(dldf,x)
print(dldx)



def numerical_convolve2d_vjp(X, W, dl__dY, h=1e-6):
  # We are going to perturb each pixel of X by a small amount h
  # and observe the corresponding change in Y
  dl__dX = np.empty_like(X)
  Y = t.conv_for(X, W) # Y before any perturbation
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      Xp = X.copy()
      Xp[i, j] += h # perturbed image at pixel (i,j)
      Yp = t.conv_for(Xp, W) # convolution result when perturbed
      # numerical derivative of image Y with respect to pixel X[i,j]
      dY__dXij = (Yp - Y) / h
      # numerical derivative of loss l with respect to pixel X[i,j]
      dl__dX[i, j] = (dl__dY * dY__dXij).sum()

  dl__dW = np.empty_like(W)
  for i in range(W.shape[0]):
    for j in range(W.shape[1]):
      Wp = W.copy()
      Wp[i, j] += h # perturbed kernel at pixel (i,j)
      Yp = t.conv_for(X, Wp) # convolution result when perturbed
      # numerical derivative of image Y with respect to pixel X[i,j]
      dY__dWij = (Yp - Y) / h
      # numerical derivative of loss l with respect to pixel X[i,j]
      dl__dW[i, j] = (dl__dY * dY__dWij).sum()

  return dl__dX, dl__dW




x = np.random.rand(28,28)
#print(x)
w = np.random.rand(7,7)
random_dl__dY = np.ones(x.shape)
num_dldx, num_dldw = numerical_convolve2d_vjp(x, w, random_dl__dY)
dldx,dldw = t.conv_vjp(random_dl__dY,x,w)


print(np.allclose(num_dldw, dldw))
print(np.allclose(num_dldx, dldx))
'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import microtorch as t
np.seterr(divide='ignore', invalid='ignore')

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
#print(train_labels)
print("Done Loading Images into test and train")

mean_image = np.array(train_imgs).mean(axis=0)
std_image = np.array(train_imgs).std(axis=0)
print(mean_image.shape, std_image.shape)

train_imgs = (np.array(train_imgs) - mean_image)/std_image
print(train_imgs.shape)
