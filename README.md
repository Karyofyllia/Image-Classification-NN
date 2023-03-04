# Image-Classification-NN
Image Classification with tensorflow keras using Neural Networks.

## Load-Edit Dataset
Download "The Street View House Numbers (SVHN)" in Local Drive > User and load it. You can find it here : "http://ufldl.stanford.edu/housenumbers/".

```
testdir = r"C:\Users\user\test_32x32"
test_data = scipy.io.loadmat(testdir)
test_labels = test_data['y']
test_images = test_data['X']

traindir = r"C:\Users\user\train_32x32"
train_data = scipy.io.loadmat(traindir)
train_labels = train_data['y']
train_images = train_data['X']
```
Shapes: <br />
**test_images**: (32, 32, 3, 26032)
**test_labels**: (26032, 1)
**train_images**: (32, 32, 3, 73257)
**train_labels**: (73257, 1)

### Edit data
```
print(test_labels[0:3])
```
![1](https://user-images.githubusercontent.com/37185221/222908180-f59cc8c8-631c-48e6-b9fa-99b300312090.PNG)

Reshape test_labels to (26032,) and train_labels to (73257,). I use [Label Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) to normalize labels 
```ruby
le = LabelEncoder()
test_labels = le.fit_transform(test_labels.ravel())
test_labels = test_labels + 1 #Dataset starts from 1 and Label Encoder starts from 0 class

print(test_labels[0:3])  
```
![2](https://user-images.githubusercontent.com/37185221/222910557-d63bba12-0f1c-430d-9b42-5d99b0e195c4.PNG)

Do the same for train_labels.
```
train_labels = le.fit_transform(train_labels.ravel())
train_labels = train_labels + 1
```

## Implement K-Neighboors Classifier
For [K Neighboors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit) I need an array of (n_samples, n_features) for training data and an array of shape (n_samples) for target data. Do the appropriate [transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html).

```ruby
train_imagesKNN, train_labelsKNN, test_imagesKNN, test_labelsKNN= train_images, train_labels, test_images, test_labels

train_imagesKNN = np.transpose(train_imagesKNN, (3,0,1,2))
print(train_imagesKNN.shape)

test_imagesKNN = np.transpose(test_imagesKNN, (3,0,1,2))
print(test_imagesKNN.shape)

train_imagesKNN = np.reshape(train_imagesKNN, (73257,3072)) # From 4D to 2D
test_imagesKNN = np.reshape(test_imagesKNN, (26032,3072))
```
![3](https://user-images.githubusercontent.com/37185221/222913458-c6612e02-bcca-47b0-a057-90df7edb17fe.PNG)

Now i have a huge image dataset of 73257 train images and 26032 test images. To decrise execution time i will choose a dataset of 1000 samples.
```
test_imagesKNNc = test_imagesKNN[0:1000,:]
train_imagesKNNc = train_imagesKNN[0:1000, :]  
test_labelsKNNc = test_labelsKNN[0:1000]
train_labelsKNNc = train_labelsKNN[0:1000]
```
Implement K Neighboor Classifier. I tested classifier for different number of neighbors and different weights (distance or uniform). I end up with the following model.
```ruby
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(train_imagesKNNc, train_labelsKNNc)
acc = knn.score(train_imagesKNNc,train_labelsKNNc)
```
Predict with accuracy metric.
```ruby
pred = knn.predict(test_imagesKNNc)
accuracy = accuracy_score(test_labelsKNNc, pred)
print("accuracy: {:.2f}%".format(accuracy * 100))
```
![4](https://user-images.githubusercontent.com/37185221/222914319-c4c4f5f2-f97e-46db-8fc9-8eb501789325.PNG)

It's obvious that the accuracy is very low, and that happens because KNN does not elaborate attributes of specific class - it finds difference of every pixel value but not features. KNN works better on e.g Tabular data

## Multi-Layer Perceptron NN
Convert RGB to Grayscale.

![3](https://user-images.githubusercontent.com/37185221/222915041-cf60952e-37ad-45d4-86b8-b050b6ed22bc.PNG)

```ruby
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
```
Now i convert arrays from `[x, y, rgb, samples]` to `[samples, x, y]`

```ruby
def images3d(data):
    images = [] #create empty list 
    for i in range(0, data.shape[3]): #73257 train images, 26032  test images
        images.append(rgb2gray(data[:, :, :, i])) 
    return np.asarray(images) #from list to numpy array
```
Normalization, data shape before and after format.
```
train_imagesNN, test_imagesNN = train_images / 255, test_images / 255 
train_labelsNN, test_labelsNN = train_labels, test_labels

print(train_imagesNN.shape ,'Train images')
print(test_imagesNN.shape, 'Test images')
print(train_labelsNN.shape, 'train labels')
print(test_labelsNN.shape, 'test labels')

train_imagesNN=images3d(train_imagesNN)
test_imagesNN=images3d(test_imagesNN)

print('----after format------')
print(train_imagesNN.shape, 'Train images')
print(test_imagesNN.shape, 'Test images')
print(train_labelsNN.shape, 'Train labels')
print(test_labelsNN.shape, 'Test labels')
```
![5](https://user-images.githubusercontent.com/37185221/222915461-97ca8120-8b4a-443d-843a-de27ca93054f.PNG)

Create Sequential NN model. 1 input layer, 3 hidden and 1 output layer.
```ruby
model = keras.models.Sequential([
keras.layers.Flatten(input_shape = [32, 32]), 
keras.layers.Dense(500, activation = 'relu' ),
keras.layers.Dense(200, activation = 'relu' ),
keras.layers.Dense(200, activation = 'relu' ),
keras.layers.Dense(200, activation = 'relu' ),
keras.layers.Dense(11, activation = 'softmax' )])

model.summary()
```
![6](https://user-images.githubusercontent.com/37185221/222915567-d5c2deb0-70a6-4482-9bef-7a472cc30a15.PNG)

Compile model using 


