
<h1> Multi-Layer Perceptrons </h1>


```python
# if you keras is not using tensorflow as backend set "KERAS_BACKEND=tensorflow" use this command
from keras.utils import np_utils 
from keras.datasets import mnist 
import seaborn as sns
from keras.initializers import RandomNormal
import tensorflow as tf
import keras


import matplotlib.pyplot as plt
import numpy as np
import time

from keras.models import Sequential 
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.layers import Dense, Activation 

%matplotlib notebook
```


```python
# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4
# https://stackoverflow.com/a/14434334
# this function is used to update the plots for each epoch and error
def plt_dynamic(x, vy, ty, ax, colors=['b']):
    ax.plot(x, vy, 'b', label="Validation Loss")
    ax.plot(x, ty, 'r', label="Train Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()
```


```python
# the data, shuffled and split between train and test sets 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

    Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
    11493376/11490434 [==============================] - 1s 0us/step
    


```python
# if you observe the input shape its 3 dimensional vector
# for each image we have a (28*28) vector
# we will convert the (28*28) vector into single dimensional vector of 1 * 784 

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]) 
```


```python
# if we observe the above matrix each cell is having a value between 0-255
# before we move to apply machine learning algorithms lets try to normalize the data
# X => (X - Xmin)/(Xmax-Xmin) = X/255

X_train = X_train/255
X_test = X_test/255
```


```python
# here we are having a class number for each image
print("Class label of first image :", y_train[0])

# lets convert this into a 10 dimensional vector
# ex: consider an image is 5 convert it into 5 => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# this conversion needed for MLPs 

Y_train = np_utils.to_categorical(y_train, 10) 
Y_test = np_utils.to_categorical(y_test, 10)

print("After converting the output into a vector : ",Y_train[0])
```

    Class label of first image : 5
    After converting the output into a vector :  [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
    


```python
# Input Parameters

epochs=20
batch=128
Input=X_train.shape[1]
Output=10

```

<h2> 1. MLP+Re-Lu+Adam optimizer(784-512-256-10) </h2>


```python
#Initialising all the layers

model1=Sequential()
model1.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model1.add(Activation('relu'))

model1.add(Dense(256,kernel_initializer='he_normal'))
model1.add(Activation('relu'))

model1.add(Dense(Output,kernel_initializer='glorot_normal'))
model1.add(Activation(tf.nn.softmax))

```


```python
model1.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 512)               401920    
    _________________________________________________________________
    activation_1 (Activation)    (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    activation_2 (Activation)    (None, 256)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                2570      
    _________________________________________________________________
    activation_3 (Activation)    (None, 10)                0         
    =================================================================
    Total params: 535,818
    Trainable params: 535,818
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Compiling the model
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

his=model1.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))

```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 13s 216us/step - loss: 0.2220 - acc: 0.9351 - val_loss: 0.1051 - val_acc: 0.9669
    Epoch 2/20
    60000/60000 [==============================] - 3s 52us/step - loss: 0.0814 - acc: 0.9747 - val_loss: 0.0831 - val_acc: 0.9749
    Epoch 3/20
    60000/60000 [==============================] - 3s 52us/step - loss: 0.0511 - acc: 0.9836 - val_loss: 0.0641 - val_acc: 0.9778
    Epoch 4/20
    60000/60000 [==============================] - 3s 55us/step - loss: 0.0340 - acc: 0.9895 - val_loss: 0.0643 - val_acc: 0.9807
    Epoch 5/20
    60000/60000 [==============================] - 3s 55us/step - loss: 0.0261 - acc: 0.9916 - val_loss: 0.0744 - val_acc: 0.9779
    Epoch 6/20
    60000/60000 [==============================] - 3s 55us/step - loss: 0.0205 - acc: 0.9935 - val_loss: 0.0704 - val_acc: 0.9796
    Epoch 7/20
    60000/60000 [==============================] - 3s 54us/step - loss: 0.0171 - acc: 0.9945 - val_loss: 0.0904 - val_acc: 0.9756
    Epoch 8/20
    60000/60000 [==============================] - 3s 53us/step - loss: 0.0161 - acc: 0.9945 - val_loss: 0.0787 - val_acc: 0.9793
    Epoch 9/20
    60000/60000 [==============================] - 3s 53us/step - loss: 0.0135 - acc: 0.9956 - val_loss: 0.0806 - val_acc: 0.9803
    Epoch 10/20
    60000/60000 [==============================] - 3s 55us/step - loss: 0.0150 - acc: 0.9948 - val_loss: 0.0790 - val_acc: 0.9789
    Epoch 11/20
    60000/60000 [==============================] - 3s 54us/step - loss: 0.0093 - acc: 0.9970 - val_loss: 0.0743 - val_acc: 0.9825
    Epoch 12/20
    60000/60000 [==============================] - 3s 54us/step - loss: 0.0105 - acc: 0.9966 - val_loss: 0.0939 - val_acc: 0.9793
    Epoch 13/20
    60000/60000 [==============================] - 3s 53us/step - loss: 0.0118 - acc: 0.9964 - val_loss: 0.0931 - val_acc: 0.9795
    Epoch 14/20
    60000/60000 [==============================] - 3s 52us/step - loss: 0.0084 - acc: 0.9972 - val_loss: 0.0828 - val_acc: 0.9828
    Epoch 15/20
    60000/60000 [==============================] - 3s 53us/step - loss: 0.0063 - acc: 0.9979 - val_loss: 0.0860 - val_acc: 0.9830
    Epoch 16/20
    60000/60000 [==============================] - 3s 52us/step - loss: 0.0100 - acc: 0.9966 - val_loss: 0.0917 - val_acc: 0.9817
    Epoch 17/20
    60000/60000 [==============================] - 3s 51us/step - loss: 0.0089 - acc: 0.9971 - val_loss: 0.0943 - val_acc: 0.9818
    Epoch 18/20
    60000/60000 [==============================] - 3s 53us/step - loss: 0.0031 - acc: 0.9990 - val_loss: 0.0850 - val_acc: 0.9826
    Epoch 19/20
    60000/60000 [==============================] - 3s 52us/step - loss: 0.0056 - acc: 0.9985 - val_loss: 0.1041 - val_acc: 0.9786
    Epoch 20/20
    60000/60000 [==============================] - 3s 51us/step - loss: 0.0123 - acc: 0.9964 - val_loss: 0.1195 - val_acc: 0.9759
    


```python
#Evaluate the accuracy and test loss

score = model1.evaluate(X_test, Y_test, verbose=0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])
```

    Test loss: 0.119451443693
    Test accuracy: 0.9759
    


```python
#Plotting the the train and test loss for each epochs

fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


![png](output_13_0.png)


<h2> 2. MLP-ReLu-Adam(784-512-256-128-10) </h2>


```python
#Initialising all the layers
model2=Sequential()

#Hidden layer_1
model2.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model2.add(Activation('relu'))

#Hidden layer_2
model2.add(Dense(256,kernel_initializer='he_normal'))
model2.add(Activation('relu'))

#hidden layer_3
model2.add(Dense(128,kernel_initializer='he_normal'))
model2.add(Activation('relu'))

#Output layer
model2.add(Dense(10,kernel_initializer='glorot_normal'))
model2.add(Activation(tf.nn.softmax))
```


```python
#Summary of the model
model2.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_4 (Dense)              (None, 512)               401920    
    _________________________________________________________________
    activation_4 (Activation)    (None, 512)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    activation_5 (Activation)    (None, 256)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 128)               32896     
    _________________________________________________________________
    activation_6 (Activation)    (None, 128)               0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 10)                1290      
    _________________________________________________________________
    activation_7 (Activation)    (None, 10)                0         
    =================================================================
    Total params: 567,434
    Trainable params: 567,434
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#Compiling the layer

model2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model2.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 4s 71us/step - loss: 0.2301 - acc: 0.9318 - val_loss: 0.1093 - val_acc: 0.9667
    Epoch 2/20
    60000/60000 [==============================] - 4s 60us/step - loss: 0.0851 - acc: 0.9729 - val_loss: 0.0820 - val_acc: 0.9744
    Epoch 3/20
    60000/60000 [==============================] - 4s 60us/step - loss: 0.0543 - acc: 0.9828 - val_loss: 0.0686 - val_acc: 0.9788
    Epoch 4/20
    60000/60000 [==============================] - 3s 58us/step - loss: 0.0388 - acc: 0.9878 - val_loss: 0.0742 - val_acc: 0.9766
    Epoch 5/20
    60000/60000 [==============================] - 4s 58us/step - loss: 0.0306 - acc: 0.9901 - val_loss: 0.0982 - val_acc: 0.9709
    Epoch 6/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0220 - acc: 0.9926 - val_loss: 0.0777 - val_acc: 0.9785
    Epoch 7/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0239 - acc: 0.9917 - val_loss: 0.0811 - val_acc: 0.9784
    Epoch 8/20
    60000/60000 [==============================] - 4s 58us/step - loss: 0.0175 - acc: 0.9942 - val_loss: 0.1047 - val_acc: 0.9740
    Epoch 9/20
    60000/60000 [==============================] - 4s 58us/step - loss: 0.0177 - acc: 0.9939 - val_loss: 0.1015 - val_acc: 0.9761
    Epoch 10/20
    60000/60000 [==============================] - 3s 56us/step - loss: 0.0145 - acc: 0.9950 - val_loss: 0.0908 - val_acc: 0.9799
    Epoch 11/20
    60000/60000 [==============================] - 3s 56us/step - loss: 0.0139 - acc: 0.9957 - val_loss: 0.0935 - val_acc: 0.9788
    Epoch 12/20
    60000/60000 [==============================] - 3s 54us/step - loss: 0.0145 - acc: 0.9950 - val_loss: 0.0924 - val_acc: 0.9781
    Epoch 13/20
    60000/60000 [==============================] - 3s 57us/step - loss: 0.0123 - acc: 0.9957 - val_loss: 0.0819 - val_acc: 0.9827
    Epoch 14/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0106 - acc: 0.9965 - val_loss: 0.1008 - val_acc: 0.9790
    Epoch 15/20
    60000/60000 [==============================] - 3s 56us/step - loss: 0.0119 - acc: 0.9962 - val_loss: 0.0945 - val_acc: 0.9791
    Epoch 16/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0110 - acc: 0.9969 - val_loss: 0.0827 - val_acc: 0.9820
    Epoch 17/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0090 - acc: 0.9969 - val_loss: 0.0847 - val_acc: 0.9832
    Epoch 18/20
    60000/60000 [==============================] - 3s 58us/step - loss: 0.0080 - acc: 0.9974 - val_loss: 0.0955 - val_acc: 0.9788
    Epoch 19/20
    60000/60000 [==============================] - 3s 58us/step - loss: 0.0125 - acc: 0.9962 - val_loss: 0.0810 - val_acc: 0.9832
    Epoch 20/20
    60000/60000 [==============================] - 3s 57us/step - loss: 0.0076 - acc: 0.9976 - val_loss: 0.0839 - val_acc: 0.9825
    


```python
#Evaluate accuracy and test loss

score=model2.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The test accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 66us/step
    The test loss is  0.0839298459997
    The test accuracy is  0.9825
    


```python
#Plotting the training and test error on each epochs

fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


![png](output_19_0.png)


<h2> 3. MLP-ReLu-Adam(784-512-256-128-64-32-10) </h2>


```python
#Initialising all Layers
model3=Sequential()

#Hidden Layer 1
model3.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model3.add(Activation('relu'))

#Hidden Layer 2
model3.add(Dense(256,kernel_initializer='he_normal'))
model3.add(Activation('relu'))

#Hidden layer 3
model3.add(Dense(128,kernel_initializer='he_normal'))
model3.add(Activation('relu'))

#Hidden Layer 4
model3.add(Dense(64,kernel_initializer='he_normal'))
model3.add(Activation('relu'))

#Hidden Layer 5
model3.add(Dense(32,kernel_initializer='he_normal'))
model3.add(Activation('relu'))

#Output Layer

model3.add(Dense(10,kernel_initializer='glorot_normal'))
model3.add(Activation(tf.nn.softmax))
```


```python
#Model Summary
model3.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_8 (Dense)              (None, 512)               401920    
    _________________________________________________________________
    activation_8 (Activation)    (None, 512)               0         
    _________________________________________________________________
    dense_9 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    activation_9 (Activation)    (None, 256)               0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    activation_10 (Activation)   (None, 128)               0         
    _________________________________________________________________
    dense_11 (Dense)             (None, 64)                8256      
    _________________________________________________________________
    activation_11 (Activation)   (None, 64)                0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 32)                2080      
    _________________________________________________________________
    activation_12 (Activation)   (None, 32)                0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                330       
    _________________________________________________________________
    activation_13 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 576,810
    Trainable params: 576,810
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#Compile
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model3.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 5s 85us/step - loss: 0.2547 - acc: 0.9232 - val_loss: 0.1063 - val_acc: 0.9692
    Epoch 2/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.0926 - acc: 0.9719 - val_loss: 0.1032 - val_acc: 0.9691
    Epoch 3/20
    60000/60000 [==============================] - 4s 71us/step - loss: 0.0598 - acc: 0.9813 - val_loss: 0.0771 - val_acc: 0.9759
    Epoch 4/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0471 - acc: 0.9851 - val_loss: 0.0695 - val_acc: 0.9792
    Epoch 5/20
    60000/60000 [==============================] - 4s 72us/step - loss: 0.0363 - acc: 0.9877 - val_loss: 0.0718 - val_acc: 0.9804
    Epoch 6/20
    60000/60000 [==============================] - 4s 72us/step - loss: 0.0303 - acc: 0.9901 - val_loss: 0.0765 - val_acc: 0.9793
    Epoch 7/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0263 - acc: 0.9914 - val_loss: 0.0680 - val_acc: 0.9819
    Epoch 8/20
    60000/60000 [==============================] - 4s 70us/step - loss: 0.0214 - acc: 0.9929 - val_loss: 0.0830 - val_acc: 0.9789
    Epoch 9/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0186 - acc: 0.9940 - val_loss: 0.0860 - val_acc: 0.9794
    Epoch 10/20
    60000/60000 [==============================] - 4s 70us/step - loss: 0.0203 - acc: 0.9936 - val_loss: 0.0862 - val_acc: 0.9797
    Epoch 11/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0214 - acc: 0.9933 - val_loss: 0.0804 - val_acc: 0.9803
    Epoch 12/20
    60000/60000 [==============================] - 4s 70us/step - loss: 0.0133 - acc: 0.9956 - val_loss: 0.0893 - val_acc: 0.9795
    Epoch 13/20
    60000/60000 [==============================] - 4s 70us/step - loss: 0.0157 - acc: 0.9953 - val_loss: 0.0959 - val_acc: 0.9777
    Epoch 14/20
    60000/60000 [==============================] - 4s 70us/step - loss: 0.0126 - acc: 0.9958 - val_loss: 0.0911 - val_acc: 0.9810
    Epoch 15/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0147 - acc: 0.9954 - val_loss: 0.0846 - val_acc: 0.9823
    Epoch 16/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0119 - acc: 0.9964 - val_loss: 0.0935 - val_acc: 0.9810
    Epoch 17/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0107 - acc: 0.9968 - val_loss: 0.0888 - val_acc: 0.9814
    Epoch 18/20
    60000/60000 [==============================] - 4s 70us/step - loss: 0.0105 - acc: 0.9966 - val_loss: 0.0984 - val_acc: 0.9801
    Epoch 19/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0103 - acc: 0.9968 - val_loss: 0.0928 - val_acc: 0.9806
    Epoch 20/20
    60000/60000 [==============================] - 4s 71us/step - loss: 0.0110 - acc: 0.9965 - val_loss: 0.0756 - val_acc: 0.9842
    


```python
#Test loss and accuracy

score=model3.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 76us/step
    The test loss is  0.075569450206
    The accuracy is  0.9842
    


```python
#plotting the train and tes loss for each epoch 
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


![png](output_25_0.png)


<h2> 4. MLP-ReLu-Dropout-Adam (784-512-DP-256-DP-10) </h2>


```python
#Initialising all layers
model4=Sequential()

#Hidden Layer 1
model4.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model4.add(Activation('relu'))

#Dropout layer
model4.add(Dropout(0.5))

#Hidden layer 2
model4.add(Dense(256,kernel_initializer='he_normal'))
model4.add(Activation('relu'))

#Dropout layer
model4.add(Dropout(0.5))

#Output Layer
model4.add(Dense(Output,kernel_initializer='glorot_normal'))
model4.add(Activation(tf.nn.softmax))

```


```python
#Model Summary
model4.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_14 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    activation_14 (Activation)   (None, 512)               0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_15 (Dense)             (None, 128)               65664     
    _________________________________________________________________
    activation_15 (Activation)   (None, 128)               0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_16 (Dense)             (None, 10)                1290      
    _________________________________________________________________
    activation_16 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 468,874
    Trainable params: 468,874
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Compile

model4.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model4.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 5s 77us/step - loss: 0.4619 - acc: 0.8573 - val_loss: 0.1450 - val_acc: 0.9552
    Epoch 2/20
    60000/60000 [==============================] - 3s 55us/step - loss: 0.2003 - acc: 0.9421 - val_loss: 0.1033 - val_acc: 0.9683
    Epoch 3/20
    60000/60000 [==============================] - 3s 58us/step - loss: 0.1568 - acc: 0.9542 - val_loss: 0.0865 - val_acc: 0.9737
    Epoch 4/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.1323 - acc: 0.9619 - val_loss: 0.0822 - val_acc: 0.9760
    Epoch 5/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.1140 - acc: 0.9654 - val_loss: 0.0747 - val_acc: 0.9771
    Epoch 6/20
    60000/60000 [==============================] - 3s 58us/step - loss: 0.1041 - acc: 0.9685 - val_loss: 0.0751 - val_acc: 0.9782
    Epoch 7/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0955 - acc: 0.9714 - val_loss: 0.0680 - val_acc: 0.9797
    Epoch 8/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0879 - acc: 0.9730 - val_loss: 0.0709 - val_acc: 0.9788
    Epoch 9/20
    60000/60000 [==============================] - 3s 58us/step - loss: 0.0841 - acc: 0.9749 - val_loss: 0.0655 - val_acc: 0.9803
    Epoch 10/20
    60000/60000 [==============================] - 4s 60us/step - loss: 0.0747 - acc: 0.9779 - val_loss: 0.0645 - val_acc: 0.9827
    Epoch 11/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0745 - acc: 0.9772 - val_loss: 0.0653 - val_acc: 0.9818
    Epoch 12/20
    60000/60000 [==============================] - 4s 62us/step - loss: 0.0701 - acc: 0.9788 - val_loss: 0.0669 - val_acc: 0.9818
    Epoch 13/20
    60000/60000 [==============================] - 3s 58us/step - loss: 0.0651 - acc: 0.9805 - val_loss: 0.0675 - val_acc: 0.9807
    Epoch 14/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0633 - acc: 0.9805 - val_loss: 0.0676 - val_acc: 0.9823
    Epoch 15/20
    60000/60000 [==============================] - 4s 61us/step - loss: 0.0610 - acc: 0.9815 - val_loss: 0.0651 - val_acc: 0.9819
    Epoch 16/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0578 - acc: 0.9824 - val_loss: 0.0599 - val_acc: 0.9838
    Epoch 17/20
    60000/60000 [==============================] - 3s 57us/step - loss: 0.0547 - acc: 0.9830 - val_loss: 0.0632 - val_acc: 0.9825
    Epoch 18/20
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0531 - acc: 0.9839 - val_loss: 0.0603 - val_acc: 0.9830
    Epoch 19/20
    60000/60000 [==============================] - 4s 60us/step - loss: 0.0505 - acc: 0.9841 - val_loss: 0.0628 - val_acc: 0.9831
    Epoch 20/20
    60000/60000 [==============================] - 3s 56us/step - loss: 0.0510 - acc: 0.9844 - val_loss: 0.0617 - val_acc: 0.9830
    


```python
# Test loss and Accuracy

score=model4.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 70us/step
    The test loss is  0.0616500246239
    The accuracy is  0.983
    


```python
#plotting the train and test loss for each epoch 
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


![png](output_31_0.png)


<h2> 5. MLP-ReLu-Dropout-Adam(784-512-Dp-256-Dp-128-Dp-10) </h2>


```python
#Initilaiisng the layer

model5=Sequential()

#Hidden Layer 1

model5.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model5.add(Activation('relu'))

#Dropout Layer
model5.add(Dropout(0.5))

#Hidden Layer 2
model5.add(Dense(256,kernel_initializer='he_normal'))
model5.add(Activation('relu'))

#Dropout Layer
model5.add(Dropout(0.5))

#Hidden Layer 3
model5.add(Dense(128,kernel_initializer='he_normal'))
model5.add(Activation('relu'))

#Dropout Layer
model5.add(Dropout(0.5))

#Output layer
model5.add(Dense(Output,kernel_initializer='glorot_normal'))
model5.add(Activation(tf.nn.softmax))

```


```python
#Model summary
model5.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_17 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    activation_17 (Activation)   (None, 512)               0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_18 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    activation_18 (Activation)   (None, 256)               0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_19 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    activation_19 (Activation)   (None, 128)               0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_20 (Dense)             (None, 10)                1290      
    _________________________________________________________________
    activation_20 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 567,434
    Trainable params: 567,434
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#Compile

model5.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model5.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 6s 94us/step - loss: 0.6169 - acc: 0.8063 - val_loss: 0.1707 - val_acc: 0.9501
    Epoch 2/20
    60000/60000 [==============================] - 4s 66us/step - loss: 0.2535 - acc: 0.9299 - val_loss: 0.1225 - val_acc: 0.9643
    Epoch 3/20
    60000/60000 [==============================] - 4s 72us/step - loss: 0.1916 - acc: 0.9468 - val_loss: 0.1059 - val_acc: 0.9701
    Epoch 4/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.1602 - acc: 0.9550 - val_loss: 0.0971 - val_acc: 0.9720
    Epoch 5/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.1429 - acc: 0.9602 - val_loss: 0.0859 - val_acc: 0.9757
    Epoch 6/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.1275 - acc: 0.9644 - val_loss: 0.0826 - val_acc: 0.9763
    Epoch 7/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.1168 - acc: 0.9674 - val_loss: 0.0761 - val_acc: 0.9783
    Epoch 8/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.1105 - acc: 0.9696 - val_loss: 0.0722 - val_acc: 0.9784
    Epoch 9/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.1012 - acc: 0.9719 - val_loss: 0.0750 - val_acc: 0.9787
    Epoch 10/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.0957 - acc: 0.9728 - val_loss: 0.0729 - val_acc: 0.9785
    Epoch 11/20
    60000/60000 [==============================] - 4s 69us/step - loss: 0.0886 - acc: 0.9750 - val_loss: 0.0678 - val_acc: 0.9810
    Epoch 12/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.0859 - acc: 0.9758 - val_loss: 0.0679 - val_acc: 0.9810
    Epoch 13/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0825 - acc: 0.9759 - val_loss: 0.0720 - val_acc: 0.9794
    Epoch 14/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.0798 - acc: 0.9777 - val_loss: 0.0696 - val_acc: 0.9808
    Epoch 15/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0745 - acc: 0.9782 - val_loss: 0.0629 - val_acc: 0.9832
    Epoch 16/20
    60000/60000 [==============================] - 4s 68us/step - loss: 0.0733 - acc: 0.9795 - val_loss: 0.0643 - val_acc: 0.9824
    Epoch 17/20
    60000/60000 [==============================] - 4s 70us/step - loss: 0.0702 - acc: 0.9794 - val_loss: 0.0624 - val_acc: 0.9814
    Epoch 18/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.0667 - acc: 0.9803 - val_loss: 0.0647 - val_acc: 0.9826
    Epoch 19/20
    60000/60000 [==============================] - 4s 70us/step - loss: 0.0651 - acc: 0.9813 - val_loss: 0.0662 - val_acc: 0.9830
    Epoch 20/20
    60000/60000 [==============================] - 4s 67us/step - loss: 0.0608 - acc: 0.9821 - val_loss: 0.0644 - val_acc: 0.9824
    


```python
#Test loss and Accuracy

score=model5.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 80us/step
    The test loss is  0.0643543138517
    The accuracy is  0.9824
    


```python
#plotting train and test loss for each epochs
 
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


![png](output_37_0.png)


<h2> 6. MLP-ReLu-Adam-Dropout (784-512-Dp-256-Dp-128-Dp-64-Dp-32-Dp-10) </h2>


```python
#Initialising all layers
model6=Sequential()

# Hidden Layer 1
model6.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model6.add(Activation('relu'))

#Dropout Layer
model6.add(Dropout(0.5))

#Hidden layer 2
model6.add(Dense(256,kernel_initializer='he_normal'))
model6.add(Activation('relu'))

#Dropout Layer
model6.add(Dropout(0.5))

#Hidden Layer 3
model6.add(Dense(128,kernel_initializer='he_normal'))
model6.add(Activation('relu'))

#Dropout Layer
model6.add(Dropout(0.5))

#Hidden layer 4
model6.add(Dense(64,kernel_initializer='he_normal'))
model6.add(Activation('relu'))

#Dropout Layer
model6.add(Dropout(0.5))

#Hidden Layer 5
model6.add(Dense(32,kernel_initializer='he_normal'))
model6.add(Activation('relu'))

#Dropout Layer
model6.add(Dropout(0.5))

#Output Layer
model6.add(Dense(Output,kernel_initializer='glorot_normal'))
model6.add(Activation(tf.nn.softmax))

```


```python
#Model summary
model6.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_21 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    activation_21 (Activation)   (None, 512)               0         
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_22 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    activation_22 (Activation)   (None, 256)               0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_23 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    activation_23 (Activation)   (None, 128)               0         
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_24 (Dense)             (None, 64)                8256      
    _________________________________________________________________
    activation_24 (Activation)   (None, 64)                0         
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_25 (Dense)             (None, 32)                2080      
    _________________________________________________________________
    activation_25 (Activation)   (None, 32)                0         
    _________________________________________________________________
    dropout_10 (Dropout)         (None, 32)                0         
    _________________________________________________________________
    dense_26 (Dense)             (None, 10)                330       
    _________________________________________________________________
    activation_26 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 576,810
    Trainable params: 576,810
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#Compile
model6.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model6.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 7s 118us/step - loss: 1.7124 - acc: 0.3829 - val_loss: 0.6910 - val_acc: 0.7342
    Epoch 2/20
    60000/60000 [==============================] - 5s 82us/step - loss: 0.8352 - acc: 0.6906 - val_loss: 0.4987 - val_acc: 0.8183
    Epoch 3/20
    60000/60000 [==============================] - 5s 82us/step - loss: 0.6374 - acc: 0.7867 - val_loss: 0.3239 - val_acc: 0.9270
    Epoch 4/20
    60000/60000 [==============================] - 5s 83us/step - loss: 0.4982 - acc: 0.8580 - val_loss: 0.2474 - val_acc: 0.9468
    Epoch 5/20
    60000/60000 [==============================] - 5s 85us/step - loss: 0.4219 - acc: 0.8890 - val_loss: 0.2008 - val_acc: 0.9527
    Epoch 6/20
    60000/60000 [==============================] - 5s 83us/step - loss: 0.3663 - acc: 0.9070 - val_loss: 0.1781 - val_acc: 0.9615
    Epoch 7/20
    60000/60000 [==============================] - 5s 84us/step - loss: 0.3252 - acc: 0.9182 - val_loss: 0.1576 - val_acc: 0.9625
    Epoch 8/20
    60000/60000 [==============================] - 5s 83us/step - loss: 0.3068 - acc: 0.9250 - val_loss: 0.1621 - val_acc: 0.9630
    Epoch 9/20
    60000/60000 [==============================] - 5s 83us/step - loss: 0.2888 - acc: 0.9291 - val_loss: 0.1430 - val_acc: 0.9673
    Epoch 10/20
    60000/60000 [==============================] - 5s 82us/step - loss: 0.2708 - acc: 0.9349 - val_loss: 0.1466 - val_acc: 0.9676
    Epoch 11/20
    60000/60000 [==============================] - 5s 87us/step - loss: 0.2533 - acc: 0.9384 - val_loss: 0.1339 - val_acc: 0.9698
    Epoch 12/20
    60000/60000 [==============================] - 5s 83us/step - loss: 0.2404 - acc: 0.9423 - val_loss: 0.1439 - val_acc: 0.9700
    Epoch 13/20
    60000/60000 [==============================] - 5s 82us/step - loss: 0.2328 - acc: 0.9446 - val_loss: 0.1333 - val_acc: 0.9722
    Epoch 14/20
    60000/60000 [==============================] - 5s 83us/step - loss: 0.2233 - acc: 0.9467 - val_loss: 0.1325 - val_acc: 0.9711
    Epoch 15/20
    60000/60000 [==============================] - 5s 80us/step - loss: 0.2197 - acc: 0.9481 - val_loss: 0.1317 - val_acc: 0.9724
    Epoch 16/20
    60000/60000 [==============================] - 5s 83us/step - loss: 0.2101 - acc: 0.9495 - val_loss: 0.1153 - val_acc: 0.9769
    Epoch 17/20
    60000/60000 [==============================] - 5s 83us/step - loss: 0.1991 - acc: 0.9519 - val_loss: 0.1301 - val_acc: 0.9745
    Epoch 18/20
    60000/60000 [==============================] - 5s 85us/step - loss: 0.2008 - acc: 0.9517 - val_loss: 0.1186 - val_acc: 0.9750
    Epoch 19/20
    60000/60000 [==============================] - 5s 86us/step - loss: 0.1904 - acc: 0.9533 - val_loss: 0.1141 - val_acc: 0.9767
    Epoch 20/20
    60000/60000 [==============================] - 5s 84us/step - loss: 0.1851 - acc: 0.9563 - val_loss: 0.1242 - val_acc: 0.9744
    


```python
#Test loss and Accuracy
score=model6.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 78us/step
    The test loss is  0.124185562286
    The accuracy is  0.9744
    


```python
#plotting the train and test loss for each epoch 
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


![png](output_43_0.png)


<h2> 7. MLP-ReLu-BN-Adam (784-512-BN-256-BN-10) </h2>


```python
#Initialising all layers
model7=Sequential()

#Hidden Layer 1
model7.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model7.add(Activation('relu'))

#Batch Normalization
model7.add(BatchNormalization())

#Hidden layer 2
model7.add(Dense(256,kernel_initializer='he_normal'))
model7.add(Activation('relu'))

#Batch Normalization
model7.add(BatchNormalization())

#Output Layer
model7.add(Dense(Output,kernel_initializer='glorot_normal'))
model7.add(Activation(tf.nn.softmax))

```


```python
#Model summary
model7.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_34 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    activation_34 (Activation)   (None, 512)               0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 512)               2048      
    _________________________________________________________________
    dense_35 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    activation_35 (Activation)   (None, 256)               0         
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 256)               1024      
    _________________________________________________________________
    dense_36 (Dense)             (None, 10)                2570      
    _________________________________________________________________
    activation_36 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 538,890
    Trainable params: 537,354
    Non-trainable params: 1,536
    _________________________________________________________________
    


```python
#Compile
model7.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model7.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 8s 127us/step - loss: 0.1811 - acc: 0.9447 - val_loss: 0.1170 - val_acc: 0.9621
    Epoch 2/20
    60000/60000 [==============================] - 6s 94us/step - loss: 0.0696 - acc: 0.9786 - val_loss: 0.0870 - val_acc: 0.9719
    Epoch 3/20
    60000/60000 [==============================] - 6s 94us/step - loss: 0.0427 - acc: 0.9864 - val_loss: 0.0700 - val_acc: 0.9778
    Epoch 4/20
    60000/60000 [==============================] - 6s 96us/step - loss: 0.0314 - acc: 0.9902 - val_loss: 0.0882 - val_acc: 0.9726
    Epoch 5/20
    60000/60000 [==============================] - 6s 94us/step - loss: 0.0252 - acc: 0.9917 - val_loss: 0.0798 - val_acc: 0.9774
    Epoch 6/20
    60000/60000 [==============================] - 6s 97us/step - loss: 0.0226 - acc: 0.9926 - val_loss: 0.0763 - val_acc: 0.9779
    Epoch 7/20
    60000/60000 [==============================] - 6s 97us/step - loss: 0.0189 - acc: 0.9934 - val_loss: 0.0755 - val_acc: 0.9778
    Epoch 8/20
    60000/60000 [==============================] - 6s 96us/step - loss: 0.0182 - acc: 0.9938 - val_loss: 0.0820 - val_acc: 0.9784
    Epoch 9/20
    60000/60000 [==============================] - 6s 94us/step - loss: 0.0164 - acc: 0.9943 - val_loss: 0.0955 - val_acc: 0.9747
    Epoch 10/20
    60000/60000 [==============================] - 6s 93us/step - loss: 0.0122 - acc: 0.9965 - val_loss: 0.0834 - val_acc: 0.9775
    Epoch 11/20
    60000/60000 [==============================] - 6s 95us/step - loss: 0.0122 - acc: 0.9961 - val_loss: 0.0919 - val_acc: 0.9763
    Epoch 12/20
    60000/60000 [==============================] - 6s 92us/step - loss: 0.0143 - acc: 0.9954 - val_loss: 0.1024 - val_acc: 0.9741
    Epoch 13/20
    60000/60000 [==============================] - 6s 94us/step - loss: 0.0114 - acc: 0.9965 - val_loss: 0.0805 - val_acc: 0.9798
    Epoch 14/20
    60000/60000 [==============================] - 6s 96us/step - loss: 0.0073 - acc: 0.9975 - val_loss: 0.0769 - val_acc: 0.9820
    Epoch 15/20
    60000/60000 [==============================] - 6s 93us/step - loss: 0.0089 - acc: 0.9970 - val_loss: 0.0780 - val_acc: 0.9813
    Epoch 16/20
    60000/60000 [==============================] - 6s 94us/step - loss: 0.0095 - acc: 0.9970 - val_loss: 0.0879 - val_acc: 0.9805
    Epoch 17/20
    60000/60000 [==============================] - 6s 94us/step - loss: 0.0099 - acc: 0.9968 - val_loss: 0.0904 - val_acc: 0.9773
    Epoch 18/20
    60000/60000 [==============================] - 6s 96us/step - loss: 0.0106 - acc: 0.9966 - val_loss: 0.0874 - val_acc: 0.9808
    Epoch 19/20
    60000/60000 [==============================] - 6s 95us/step - loss: 0.0085 - acc: 0.9970 - val_loss: 0.0833 - val_acc: 0.9803
    Epoch 20/20
    60000/60000 [==============================] - 6s 96us/step - loss: 0.0067 - acc: 0.9978 - val_loss: 0.0739 - val_acc: 0.9817
    


```python
#Test loss and Accuracy
score=model7.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 86us/step
    The test loss is  0.0738838885449
    The accuracy is  0.9817
    


```python
#Plotting the train and test error for each epochs
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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vn37uN1KeZ3y2lxtBMn8kvHNN98EO8yLLrroFHahAFABYHNL0h1A9RPLaiUO93cZKJgWm/JqWqLpxvv999/lo48+kvnnn1+4Q2OCCIr4Qmc+Vh8AoPIaz2q4w/zxxx/LggsuKF26dJmskwJABYDxrKj1q3QHMKsEHevvE0jQHUDHjM/gdFy24wgA1nsppxWBAsC0knO7X1a9NrI1BYAKALNavwLArBJ0rL/LL04bovKJX+XVhgUlH1MBYHKZVffICoqy3T3f3ll5VQDYWF+aBzCbPSsAzCY/53r7BBJ0B9A58zM2IZftWAGgyG677SaUw3//+99A52uvvbasvvrqcv7557dqA/PMM48cd9xxcuCBB2Y67o7GOeigg4zZm62BFADakmw4rgLAbPJVAJhNfs71dvnFaUNYPvGrvNqwoORjlhUAbrHFFkGk8RNPPDEF0//73/9kzTXXlGHDhsmKK67YVCi1APC7774L0tdMN910xgDg9ddfHwDG0aNHTzYmAyOmmWYaY/6X9SZMGW200Uby888/B4E5aUkBYFrJxeunADCenFq7SgFgNvk519snkEDh+8Sv8urGcisrAHzwwQdlm222mRTAUi3NffbZR15++WV55ZVXYgm5FgDG6ZR0B7A1ABjnXlmvUQCYVYL59FcAmE3OdgDgffeJ3H+/yMYbi+y5Z7YZGuitL04DQnR0CNWto4rJOC2X9VpWAMik1QRhBxxwgJxyyimTNMRI0znmmEPOPPNM4bEqZb/ffvvJU089JV999ZXMN998wecHH3zwpD7NjoC//PJL2XvvveXJJ58M8thx7KOOOmqyI+Brr71WbrnlFvnwww9llllmka222krOOeecYHcvAmDVZnT66afLv//974AH7gxGR8AtLS1yyCGHBPdi4uRNN91ULrvssiBFD4l9HnvssWD+J598svz444+y2WabyTXXXNPq7l4zAMidvVNPPVUIUrlDudRSSwVz564haezYsXLYYYcJQff3338fyJdH38ccc4wwxQvncfPNNwfynXXWWWWHHXaQiy66aIpVoz6AjR8kCgCzPWjtAEAsDOndWwTfKgWLvGhy+WViWjY+8UrZ+cSv8mp6taQbr95LGe90JIVONx57ZTkqZCaaVopFTDEhApB77703AF1R7kICEQI+JraeaaaZAt++s88+WzbffPMAmA0ZMiT4+2233RbsIJKaAcCN8eX/66+/DkAW09oQoL366qty3nnnTfIBvPHGG4Pj5gUWWEA++OCDAJj+3//9n1x66aVBDrzLL79c/vOf/8ibb74Z3JPHywSH1QCQclt++eVl5plnlgsvvDDox3E47+iomwDwkksuCYAhgde3334bAK79998/AHH1qBkAJB+cG0HscsstJ9ddd10AOkeOHCkLLbRQIL+rr746ALgzzjhjAAIp35122knuuuuu4N533323dO/ePfh8xIgR8q9//UsBYMIlpAAwocBqLrcDAG+4QfD1T7DiRPr1yzZDA731xWlAiI4Oobp1VDEZp+WyXusBwF9/FewmZWQ6ZXfkjwYwitf57bffDkAHd/c22GCDoNN6660nc889t9xxxx2tDkIAyJ0zgpdmAPCtt94KdsR4pLzSSisF1xPgLLPMMgFIai0I5M4775TDDz9cuHtIau0IuBoA9u/fX7bcckvhLiB5IL3++usBKBs+fLissMIKwQ4gASDHJYAkHXHEETJ06NAA3NajZgCwW7ducuSRRwY7ehERzK6zzjrBvcjj+++/L5wf/Qir8zuee+650qdPn2Ce3LFsRLoD2NiuFQDGW/etXWUHAA4YILLJJiJLLy3yxhvZZmigt8svEwPsTTaET7yScZ/4VV5Nr5Z045UZAJLjtdZaK9iluvXWW4OdN1aZGIBn9oYbbjhJIFdeeaVwh45JiBk4wp21lVdeWZ5//vngmkY7gPfBBWiXXXYJdhKrK6QQBPEoOAKABGA8NiUoJbhkxQv2YevcuXMsAMhdv6uuuiqozFJN3C3k7iPnQQD48MMPy2uvvTbpEu7gcdfu3XffTQwAGfAS7YxSlhHxiPmdd94JZPnSSy/BA2pjmX322QOgzZ1T/p9EmbIfd0a549mjRw9hgA4ry9SSAkAFgOmeUvF62QGA+AaIr4CCvW/B3ne8mVi8Sl+cFoVb8NCq24IVYOn2Luu1zEfAVBeBHf3nuCPG3ajbb799siNh7gTSf4/garXVVguOXnmkySNc7uo1A4BMDUOASOBYDQA5zllnnRUAwDewMcCxe/XqFRzH8uj5mWeekX333XdS5G2cHcALLrggAHq1QI734vHszjvvPMkHMJo75890NTyi5S5dPWq0A8gjZPrtPffcc0HkdESUKYHo448/HnxEUPvoo48G/od9+/YNjqCj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" width="640">


<h2> 8. MLP-ReLu-BN-Adam (784-512-BN-256-BN-128-BN-10) </h2>


```python
#Initilaiisng the layer

model8=Sequential()

#Hidden Layer 1

model8.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model8.add(Activation('relu'))

#Batch Normalization Layer
model8.add(BatchNormalization())

#Hidden Layer 2
model8.add(Dense(256,kernel_initializer='he_normal'))
model8.add(Activation('relu'))

#Batch Normalization Layer
model8.add(BatchNormalization())

#Hidden Layer 3
model8.add(Dense(128,kernel_initializer='he_normal'))
model8.add(Activation('relu'))

#Batch Normalization Layer
model8.add(BatchNormalization())

#Output layer
model8.add(Dense(Output,kernel_initializer='glorot_normal'))
model8.add(Activation(tf.nn.softmax))

```


```python
#Model Summary
model8.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_37 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    activation_37 (Activation)   (None, 512)               0         
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 512)               2048      
    _________________________________________________________________
    dense_38 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    activation_38 (Activation)   (None, 256)               0         
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 256)               1024      
    _________________________________________________________________
    dense_39 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    activation_39 (Activation)   (None, 128)               0         
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 128)               512       
    _________________________________________________________________
    dense_40 (Dense)             (None, 10)                1290      
    _________________________________________________________________
    activation_40 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 571,018
    Trainable params: 569,226
    Non-trainable params: 1,792
    _________________________________________________________________
    


```python
#Compile
model8.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model8.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 10s 166us/step - loss: 0.1820 - acc: 0.9451 - val_loss: 0.1067 - val_acc: 0.9668
    Epoch 2/20
    60000/60000 [==============================] - 7s 119us/step - loss: 0.0701 - acc: 0.9788 - val_loss: 0.0862 - val_acc: 0.9726
    Epoch 3/20
    60000/60000 [==============================] - 7s 119us/step - loss: 0.0466 - acc: 0.9850 - val_loss: 0.0846 - val_acc: 0.9736
    Epoch 4/20
    60000/60000 [==============================] - 7s 117us/step - loss: 0.0351 - acc: 0.9888 - val_loss: 0.0792 - val_acc: 0.9754
    Epoch 5/20
    60000/60000 [==============================] - 7s 118us/step - loss: 0.0284 - acc: 0.9906 - val_loss: 0.0774 - val_acc: 0.9772
    Epoch 6/20
    60000/60000 [==============================] - 7s 122us/step - loss: 0.0245 - acc: 0.9918 - val_loss: 0.0777 - val_acc: 0.9765
    Epoch 7/20
    60000/60000 [==============================] - 7s 122us/step - loss: 0.0224 - acc: 0.9923 - val_loss: 0.0727 - val_acc: 0.9788
    Epoch 8/20
    60000/60000 [==============================] - 7s 117us/step - loss: 0.0194 - acc: 0.9937 - val_loss: 0.0751 - val_acc: 0.9785
    Epoch 9/20
    60000/60000 [==============================] - 7s 118us/step - loss: 0.0187 - acc: 0.9939 - val_loss: 0.0787 - val_acc: 0.9788
    Epoch 10/20
    60000/60000 [==============================] - 7s 118us/step - loss: 0.0151 - acc: 0.9950 - val_loss: 0.0660 - val_acc: 0.9821
    Epoch 11/20
    60000/60000 [==============================] - 7s 117us/step - loss: 0.0151 - acc: 0.9949 - val_loss: 0.0743 - val_acc: 0.9800
    Epoch 12/20
    60000/60000 [==============================] - 7s 118us/step - loss: 0.0151 - acc: 0.9950 - val_loss: 0.0913 - val_acc: 0.9758
    Epoch 13/20
    60000/60000 [==============================] - 7s 116us/step - loss: 0.0156 - acc: 0.9948 - val_loss: 0.0902 - val_acc: 0.9753
    Epoch 14/20
    60000/60000 [==============================] - 7s 116us/step - loss: 0.0097 - acc: 0.9968 - val_loss: 0.0879 - val_acc: 0.9785
    Epoch 15/20
    60000/60000 [==============================] - 7s 117us/step - loss: 0.0108 - acc: 0.9962 - val_loss: 0.0798 - val_acc: 0.9802
    Epoch 16/20
    60000/60000 [==============================] - 7s 118us/step - loss: 0.0097 - acc: 0.9966 - val_loss: 0.0733 - val_acc: 0.9798
    Epoch 17/20
    60000/60000 [==============================] - 7s 116us/step - loss: 0.0100 - acc: 0.9966 - val_loss: 0.0778 - val_acc: 0.9814
    Epoch 18/20
    60000/60000 [==============================] - 7s 117us/step - loss: 0.0078 - acc: 0.9973 - val_loss: 0.0763 - val_acc: 0.9802
    Epoch 19/20
    60000/60000 [==============================] - 7s 119us/step - loss: 0.0076 - acc: 0.9976 - val_loss: 0.0774 - val_acc: 0.9812
    Epoch 20/20
    60000/60000 [==============================] - 7s 112us/step - loss: 0.0106 - acc: 0.9968 - val_loss: 0.0886 - val_acc: 0.9793
    


```python
#Test loss and Accuracy
score=model8.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 111us/step
    The test loss is  0.0886129535915
    The accuracy is  0.9793
    


```python
#Plotting the train and test error for each epochs
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAgAElEQVR4XuxdBdhcxdU+MSQ4BIcQJDjFi2uLBXdLIbhrsUILQQo/7kWLE6xIoAQJEqBAcYo7wS04BAlJ/ve9+91ks9lv98qcu3N3zjzPybfZvTP32Nx9d2bOOV3EmmnANGAaMA2YBkwDpgHTQFAa6BKUtCasacA0YBowDZgGTAOmAdOAGAA0JzANmAZMA6YB04BpwDQQmAYMAAZmcBPXNGAaMA2YBkwDpgHTgAFA8wHTgGnANGAaMA2YBkwDgWnAAGBgBjdxTQOmAdOAacA0YBowDRgANB8wDZgGTAOmAdOAacA0EJgGDAAGZnAT1zRgGjANmAZMA6YB04ABQPMB04BpwDRgGjANmAZMA4FpwABgYAY3cU0DpgHTgGnANGAaMA0YADQfMA2YBkwDpgHTgGnANBCYBgwABmZwE9c0YBowDZgGTAOmAdOAAUDzAdOAacA0YBowDZgGTAOBacAAYGAGN3FNA6YB04BpwDRgGjANGAA0HzANmAZMA6YB04BpwDQQmAYMAAZmcBPXNGAaMA2YBkwDpgHTgAFA8wHTgGnANGAaMA2YBkwDgWnAAGBgBjdxTQOmAdOAacA0YBowDRgANB8wDZgGTAOmAdOAacA0EJgGDAAGZnAT1zRgGjANmAZMA6YB04ABQPMB04BpwDRgGjANmAZMA4FpwABgYAY3cU0DpgHTgGnANGAaMA0YADQfMA2YBkwDpgHTgGnANBCYBgwABmZwE9c0YBowDZgGTAOmAdOAAUDzAdOAacA0YBowDZgGTAOBacAAYGAGN3FNA6YB04BpwDRgGjANGAA0HzANmAZMA6YB04BpwDQQmAYMAAZmcBPXNGAaMA2YBkwDpgHTgAFA8wHTgGnANGAaMA2YBkwDgWnAAGBgBjdxTQOmAdOAacA0YBowDRgANB8wDZgGTAOmAdOAacA0EJgGDAAGZnAT1zRgGjANmAZMA6YB04ABQPMB04BpwDRgGjANmAZMA4FpwABgYAY3cU0DpgHTgGnANGAaMA0YADQfMA2YBkwDpgHTgGnANBCYBgwABmZwE9c0YBowDZgGTAOmAdOAAUDzAdOAacA0YBowDZgGTAOBacAAYGAGN3FNA6YB04BpwDRgGjANGAA0HzANmAZMA6YB04BpwDQQmAYMAAZmcBPXNGAaMA2YBkwDpgHTgAFA8wHTgGnANGAaMA2YBkwDgWnAAGBgBjdxTQOmAdOAacA0YBowDRgANB8wDZgGTAOmAdOAacA0EJgGDAAGZnAT1zRgGjANmAZMA6YB04ABQPMB04BpwDRgGjANmAZMA4FpwABgYAY3cU0DpgHTgGnANGAaMA0YADQfMA2YBkwDpgHTgGnANBCYBgwABmZwE9c0YBowDZgGTAOmAdOAAUDzAdOAacA0YBowDZgGTAOBacAAYGAGN3FNA6YB04BpwDRgGjANGAA0HzANmAZMA6YB04BpwDQQmAYMAAZmcBPXNGAaMA2YBkwDpgHTgAFA8wHTgGnANGAaMA2YBkwDgWnAAGBgBjdxTQOmAdOAacA0YBowDRgANB8wDZgGTAOmAdOAacA0EJgGDAAGZnAT1zRgGjANmAZMA6YB04ABQPMB04BpwDRgGjANmAZMA4FpwABgPoNTf7OBvs83jPU2DZgGTAOmAdOAaaBgDUyF+30MGlvwfb24nQHAfGaYHd0/zDeE9TYNmAZMA6YB04BpoEUamAP3/ahF927pbQ0A5lP/1Oj+7QcffCBTT82X7dlGjRol9957r6y99trSo0eP9hSyQ6qQZKXIIclrsrbv1DXbtqdtNe363XffyZxzzknFTQP6rj012FgqA4D5rB4BQLS2B4BDhgyRfv36BQEAQ5E1BoChyMsvE5M13wPP195mW18tk48vTbsSAE4zDbGfAcB8Vgq3twHANrO95gPHR1WFJK/J6qMHuuHJbOtGj76NomlXA4AitgKYz+MNAObTn3e9NR843gnbsQVsq2I+WiYfT+bH+fTnc++QbKspqwFAA4B557kBwLwa9Ky/5gPHM1EjdkKS12T10QPd8GS2daNH30bRtKsBQAOAef3dAGBeDXrWX/OB45moBgB9NIgjnsrgx2PHjpXffvtNRo8enVtqyvvwww/LqquuGsQ5ZZO1uct069ZNunfvLl261N/oNABoALC5FzW+wgBgXg161r8MX5wuVRaSvCarS8/JN9avv/4qn3zyiYwcOTLfQB29CSZ/+uknmXzyyTv9wndyIw8GMVmTG6Fnz54y66yzyiSTTDJRJwOABgCTe1L9Kw0A5tWgZ/1DAgm2BeyZ8zlkx2c/HjNmjLz55pvCFZoZZ5wx+nLubJUmqUo45g8//CBTTjmldO3aNWm3Ul5nsjY3G0Eyf2R88cUX0Qpz3759J/ILA4AGAJt7kq0A2jmxvF7icX+fgYJrtZmsrjWabbyff/5Z3n33XZlrrrmEKzQuGkERv9CZjzUEAGiyJvMarjC/9957Mvfcc8tkk002QScDgAYAk3lR51fZCmBeDXrWPySQYCuAnjmfQ3Z89uMYANb7Us6qAgOAWTXnd7+8dm3kawYADQDm9X4DgHk16Fl/n784NVQVkrwmq4YHpR/TAGB6nVX3yAuK8t292N55ZTUA2Nhelgcwnz8bAMynP+96hwQSbAXQO/dzxpDPfmwAUKR///5CPfzrX/+KbL7yyivL8ssvL6eddlqnPjDHHHPIEUccIXvvvXeu7e54nH333deZv2kNZABQS7OVcQ0A5tOvAcB8+vOut89fnBrKCklek1XDg9KPWVYAuOGGG0aRxvfdd99EQj/++OOy4ooryjPPPCNLLbVUU6XUAsCvvvoqSl8z1VRTOQOAl156aQQYR4wYMcGYDIyYYoopnJ2/rMcwdbTWWmvJ999/HwXmZG0GALNqLlk/A4DJ9NTZVQYA8+nPu94hgQQqPyR5TVY/pltZAeBtt90mm2222bgAlmpt7rbbbvL000/Lc889l0jJtQAwSae0K4CdAcAk98p7jQHAvBospr8BwHx61gGAN98scsstImuvLbLjjvk4dNDbvjgdKNHTIcy2nhomJ1s+27WsAJBJqwnC9tprLznmmGPGWYiRprPMMouceOKJwm1V6n6PPfaQBx54QD777DPp3bt39P5+++03rk+zLeBPP/1Udt11V7n//vujPHYc+5BDDplgC/jiiy+Wq666St555x2ZYYYZZOONN5aTTz45Wt2LAVi1Gx1//PHy17/+NZKBK4PxFvDw4cNl//33j+7FxMnrrbeenHvuuVGKHjb2ufvuuyP+jz76aPn2229l/fXXl4suuqjT1b1mAJAre8cee6wQpHKFcpFFFol456oh2y+//CIHHnigEHR//fXXkX659X3YYYcJU7yQjyuvvDLSb69evWSrrbaSM888c6JZY2cAGz9IDADme9DqAEBMDBk4UPAEELnkknwcOujt85eJA/EmGCIkWSl4SPKarK5nS7bx6n0p4zsdSaGzjcdeebYKmYmmk2IREzFEAHLTTTdFoCvOXUggQsDHxNbTTTdddLbv//7v/2SDDTaIgNl//vOf6PNrrrkmWkFkawYA18aP/88//zwCWUxrQ4D2/PPPy6mnnjruDOBll10WbTf36dNH3n777QiYrrvuunLOOedEOfDOO+88+fvf/y4vv/xydE9uLxMcVgNA6m2JJZaQ6aefXs4444yoH8ch3/FWNwHg2WefHQFDAq8vv/wyAlx77rlnBOLqtWYAkHKQN4LYxRdfHF9zl0Sg89VXX5V55pkn0t+FF14YAdxpp502AoHU7zbbbCPXX399dO8bbrhBFlpooej9l156SXbZZRcDgCmnkAHAlAqruVwHAP7znxXwh8ksd92Vj0MHve2L04ESPR3CbOupYXKy5bNd6wHAH38UrCblFDpjd+SPBjBK1vm1116LQAdX99ZYY42o02qrrSazzz67DBo0qNNBCAC5ckbw0gwAvvLKK9GKGLeUl1566eh6ApzFFlssAkmdBYFcd911ctBBBwlXD9k62wKuBoB34ftlo402Eq4CUga2F154IQJlzz77rCy55JLRCiABIMclgGQ7+OCD5cknn4zAbb3WDADOPPPM8uc//zla0Ysbwewqq6wS3YsyvvXWW/j6uys6R1id3/GUU06RK664IuKTK5aNmq0ANvZrA4DJ5n1nV+kAwHvvFVlnHZFFFxV58cV8HDro7fOXiQPxJhgiJFkpeEjymqyuZ0u28coMACnxSiutFK1SXX311dHKG6tM3Itn9h//+MdxCvnHP/4hXKFjEmIGjnBlbZlllpHHHnssuqbRCuDNOAK03XbbRSuJ1RVSCIK4FRwDQAIwbpsSlBJcsuIF+5AmnXTSRACQq34XXHBBVJmlunG1kKuP5IMA8I477pD//e9/4y7hCh5X7d54443UAJABL/HKKHUZN24xv/7665Eun3rqKZyAWltmmmmmCGhz5ZT/Z6NO2Y8ro1zx7NevnzBAh5VlapsBQAOA2Z5SyXrpAED8AsRPQMHat2DtOxknilfZF6eicls8tNm2xQZQur3Pdi3zFjDNRWDH83NcEeNq1LXXXjvBljBXAnl+j+BqueWWi7ZeuaXJLVyu6jUDgEwNQ4BI4FgNADnOSSedFAHAF7EwwLH32WefaDuWW88PPfSQ7L777uMib5OsAJ5++ukR0KsFcrwXt2e33XbbcWcAY97JP9PVcIuWq3T1WqMVQG4h89zeo48+GkVOx406JRC95557orcIau+8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RjA0PSpDcdVts802Sw0A77jjDtliiy3kvffei/IwsnGlb+mll45WGQn0COyuRODh1lszLnPCNv/888suu+wihx9+eCpd5bnYAGAe7TXvawCwuY4aXaELAPmrmL9O8UtN3kJwM5bcW9F8/jJxrY+QZKXuQpLXZHU9W7KNFwIA5MoegVXcmISYK3k33XSTfPTRR8IExb/88otsi2T/V13FJBf1VwCPOuqoaNs1btyS3XnnnSNwmXYL+JRTTpHLL79cXn311QkMxyIGBLRbbbWVHHHEEdHqIFccueLH91ixhe0fCEQ84IADopVKfkYwyZVBzWYAUFO7lRq41rJrQBcAki/86pI33xQZNgzZCZuVRc4uSKOe9sWpo1cfRjXb+mAF9zz4bNcQtoAJshZccHwRq+OOO07OP/98Oeuss2ThhRfGguMUSO6wV7SFGkfs1tsC5pk/btPGjWMy2ph90wLAk08+OVrde+UVptgd37jqRxC65ZZbRm+Sd+bLJPHMIs8zrr/++tFnXD28E2fT70FxgrvuukvOO+882X13Zm3TaQYAdfQaj2oAMJ9+9QEgDg/jxC4S1iBjzfZMW1h88/nLxLU2QpKVugtJXpPV9WzJNl4IQSC1AJB11LmFShDIxhVBnrcj6CsKADbaAn7xxRejLeDatummm0bBIjfeeONEnzFYhACRgS5azQCglmYr4xoAzKdffQDICiBXXy1y0klIZV2byzof80l72xdnUk2V7zqzbflsloRjn+0aIgDkit29994bBXIwCISrcQwKWW+99ZwDQEYm1yaTZgQxgz0YBMLzf9zm/QnnyquDQL799ttxkclzzTWXvP/++9K/f38ZgIIExx57rOy7776y8cYbR8D1S5xP58rfYostFq0qajUDgFqaNQDoQrP6AJAVQAj+9tlHsN7ugufUY/j8ZZJamCYdQpKVqghJXpPV9WzJNl6IAPALZHHYaaedcJJnWAQA99577yiKls31CiBzFNY2nhlkYArTwPBMIfno3r17tLUbp4EZiShs8sg0L59//rnMOOOM0bYwwSoTVHOMoUOHyscffxzlO2QKGaa6mXbaabM5QoJeBgATKCnHJbYCmEN56KoPAFkBhOAPv7zwkzEftxl72xdnRsWVoJvZtgRGysCiz3ZtBwCYwSTOuuQFRc4YKWCgvLJaGpjGRjIAmM+J9QEgQvdl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" width="640">


<h2> 9. MLP-ReLu-BN-Adam(784-512-BN-256-BN-128-BN-64-BN-32-BN-10)</h2>


```python
#Initialising all layers
model9=Sequential()

# Hidden Layer 1
model9.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model9.add(Activation('relu'))

#Batch Normalization Layer
model9.add(BatchNormalization())

#Hidden layer 2
model9.add(Dense(256,kernel_initializer='he_normal'))
model9.add(Activation('relu'))

#Batch Normalization Layer
model9.add(BatchNormalization())

#Hidden Layer 3
model9.add(Dense(128,kernel_initializer='he_normal'))
model9.add(Activation('relu'))

#Batch Normalization Layer
model9.add(BatchNormalization())

#Hidden layer 4
model9.add(Dense(64,kernel_initializer='he_normal'))
model9.add(Activation('relu'))

#Batch Normalization Layer
model9.add(BatchNormalization())

#Hidden Layer 5
model9.add(Dense(32,kernel_initializer='he_normal'))
model9.add(Activation('relu'))

#Batch Normalization Layer
model9.add(BatchNormalization())

#Output Layer
model9.add(Dense(Output,kernel_initializer='glorot_normal'))
model9.add(Activation(tf.nn.softmax))

```


```python
#Model Summary
model9.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_41 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    activation_41 (Activation)   (None, 512)               0         
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 512)               2048      
    _________________________________________________________________
    dense_42 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    activation_42 (Activation)   (None, 256)               0         
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 256)               1024      
    _________________________________________________________________
    dense_43 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    activation_43 (Activation)   (None, 128)               0         
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 128)               512       
    _________________________________________________________________
    dense_44 (Dense)             (None, 64)                8256      
    _________________________________________________________________
    activation_44 (Activation)   (None, 64)                0         
    _________________________________________________________________
    batch_normalization_11 (Batc (None, 64)                256       
    _________________________________________________________________
    dense_45 (Dense)             (None, 32)                2080      
    _________________________________________________________________
    activation_45 (Activation)   (None, 32)                0         
    _________________________________________________________________
    batch_normalization_12 (Batc (None, 32)                128       
    _________________________________________________________________
    dense_46 (Dense)             (None, 10)                330       
    _________________________________________________________________
    activation_46 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 580,778
    Trainable params: 578,794
    Non-trainable params: 1,984
    _________________________________________________________________
    


```python
#Compile
model9.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model9.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 13s 217us/step - loss: 0.2567 - acc: 0.9294 - val_loss: 0.1192 - val_acc: 0.9629
    Epoch 2/20
    60000/60000 [==============================] - 9s 156us/step - loss: 0.0913 - acc: 0.9723 - val_loss: 0.0954 - val_acc: 0.9703
    Epoch 3/20
    60000/60000 [==============================] - 9s 153us/step - loss: 0.0639 - acc: 0.9800 - val_loss: 0.0954 - val_acc: 0.9715
    Epoch 4/20
    60000/60000 [==============================] - 9s 155us/step - loss: 0.0517 - acc: 0.9841 - val_loss: 0.0931 - val_acc: 0.9716
    Epoch 5/20
    60000/60000 [==============================] - 9s 156us/step - loss: 0.0415 - acc: 0.9867 - val_loss: 0.0876 - val_acc: 0.9736
    Epoch 6/20
    60000/60000 [==============================] - 9s 154us/step - loss: 0.0345 - acc: 0.9887 - val_loss: 0.0735 - val_acc: 0.9778
    Epoch 7/20
    60000/60000 [==============================] - 9s 156us/step - loss: 0.0335 - acc: 0.9894 - val_loss: 0.0736 - val_acc: 0.9763
    Epoch 8/20
    60000/60000 [==============================] - 10s 159us/step - loss: 0.0273 - acc: 0.9912 - val_loss: 0.0870 - val_acc: 0.9764
    Epoch 9/20
    60000/60000 [==============================] - 10s 159us/step - loss: 0.0264 - acc: 0.9914 - val_loss: 0.1053 - val_acc: 0.9707
    Epoch 10/20
    60000/60000 [==============================] - 10s 159us/step - loss: 0.0215 - acc: 0.9931 - val_loss: 0.0739 - val_acc: 0.9796
    Epoch 11/20
    60000/60000 [==============================] - 9s 156us/step - loss: 0.0186 - acc: 0.9938 - val_loss: 0.0879 - val_acc: 0.9762
    Epoch 12/20
    60000/60000 [==============================] - 9s 158us/step - loss: 0.0194 - acc: 0.9932 - val_loss: 0.0893 - val_acc: 0.9751
    Epoch 13/20
    60000/60000 [==============================] - 10s 165us/step - loss: 0.0196 - acc: 0.9935 - val_loss: 0.0774 - val_acc: 0.9784
    Epoch 14/20
    60000/60000 [==============================] - 9s 157us/step - loss: 0.0181 - acc: 0.9943 - val_loss: 0.0746 - val_acc: 0.9796
    Epoch 15/20
    60000/60000 [==============================] - 10s 160us/step - loss: 0.0172 - acc: 0.9947 - val_loss: 0.0701 - val_acc: 0.9808
    Epoch 16/20
    60000/60000 [==============================] - 9s 156us/step - loss: 0.0144 - acc: 0.9954 - val_loss: 0.0724 - val_acc: 0.9799
    Epoch 17/20
    60000/60000 [==============================] - 9s 158us/step - loss: 0.0131 - acc: 0.9956 - val_loss: 0.0771 - val_acc: 0.9800
    Epoch 18/20
    60000/60000 [==============================] - 9s 157us/step - loss: 0.0154 - acc: 0.9948 - val_loss: 0.0819 - val_acc: 0.9786
    Epoch 19/20
    60000/60000 [==============================] - 9s 157us/step - loss: 0.0134 - acc: 0.9955 - val_loss: 0.0795 - val_acc: 0.9799
    Epoch 20/20
    60000/60000 [==============================] - 9s 157us/step - loss: 0.0098 - acc: 0.9967 - val_loss: 0.1132 - val_acc: 0.9764
    


```python
#Test loss and Accuracy
score=model9.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 132us/step
    The test loss is  0.113197082691
    The accuracy is  0.9764
    


```python
#Plotting the train and test error for each epochs
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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JQAkAGztZfmfsAWAWsdRLrlE5JhjRI7F9cVyWsgvE98aiUlWAkDf3hPOeCH7cVUBYKsE0zvssINceeWVuZ3gu+++q+VFnHhixCh23DoDRVu1lQ29AVWkukAjALQ1Iu8AuunXFgAed5ymnda801oUXC5DesJyWsgvE98aiUlWAkDf3hPOeCH7cVUBIJIJ19uNN96o38uPkXfeeWfM7yaYYAKZbDLkFB67wRY+KygRAKZfZ9wB7FxXBIDpfam9J20B4D/+IbLLLiJrry1y771unDr0Dvll4iBWu11jkpUA0Lf3hDNeyH5cVQDYaF3s9O2///7yww8oI/97e/vtt2X++eeXW265Rc7UUp7PPvtsbVdwtdVWk3322Uee0Pyu33//vcw999w1ALnpppuO6dx8BIxqKQcddJC8+uqrcuutt9bKlvXr10+23357QRkzlB/t3r37WPO32gH88MMPZd9995VHHnmkBkr79Okj55xzTr0kmrzwwgtywAEHyIsvvlgbe95559W9h8tk0UUXlQ8++KAmw5NPPlkLHptzzjnljDPOkDXWQJ0Fm8YdQBu91kclAHTTry0AvO8+kXXWEVl4YS1i11kVOzchWvUO+WXSivesf49JVgLArN5RnedD9uN2AaDW8hUtzZW3OQEFlPHMmGe1FQCca6655LTTTpNFFllEsDMImW+//fbaHT/UK77jjjvkkEMOkeeff14TPCQZHtoDgCNHjpT+/fvLqquuKig5evzxxwtA5lRTTZUZAGIs8DPttNPK6aefXuNp9913l+mnn17uw7tGG/heeeWVa7zhyPslzUCx0EILyYILLlgDeqinizuKkOmNN96o8bGCliu1ak52Vaa4A9i5ZQgA3TzXFgC+9proihWZckqRb79149Shd8gvEwex2u0ak6wEgL69J5zxQvbjdl/KP/+s6fk7v/tmpt2ftDbARBNlGr4VALzoootkN9zh7qStvvrqstxyy8kJJ5zQIQBcf/315dJLL639HWBoSn0XYNdtk002yQwA77rrLtlss83k448/rtViRsNO35JLLlnbZQTQA7C76qqrZMstUXth7DbPPPPobaSd5dBDD82kK5eHCQBdtNe6LwFgax119oQtANRLwfoVK5kf3451cZbRQn6Z+NZHTLISAPr2nnDGC9mPYwCA2NkDsKq3ESNG1HbybtIa74MHD5bf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" width="640">


<h2> 10. MLP-ReLu-BN-Dropout-Adam (784-512-BN-DP-256-BN-DP-10) </h2>


```python
#Initialising all layers
model10=Sequential()

#Hidden Layer 1
model10.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model10.add(Activation('relu'))

#Batch Normalization layer
model10.add(BatchNormalization())

#Dropout layer
model10.add(Dropout(0.5))

#Hidden layer 2
model10.add(Dense(256,kernel_initializer='he_normal'))
model10.add(Activation('relu'))

#Batch Normalization layer
model10.add(BatchNormalization())

#Dropout layer
model10.add(Dropout(0.5))

#Output Layer
model10.add(Dense(Output,kernel_initializer='glorot_normal'))
model10.add(Activation(tf.nn.softmax))

```


```python
#Model Summary
model10.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_47 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    activation_47 (Activation)   (None, 512)               0         
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 512)               2048      
    _________________________________________________________________
    dropout_11 (Dropout)         (None, 512)               0         
    _________________________________________________________________
    dense_48 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    activation_48 (Activation)   (None, 256)               0         
    _________________________________________________________________
    batch_normalization_14 (Batc (None, 256)               1024      
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 256)               0         
    _________________________________________________________________
    dense_49 (Dense)             (None, 10)                2570      
    _________________________________________________________________
    activation_49 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 538,890
    Trainable params: 537,354
    Non-trainable params: 1,536
    _________________________________________________________________
    


```python
#Compile
model10.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model10.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 10s 168us/step - loss: 0.4150 - acc: 0.8751 - val_loss: 0.1359 - val_acc: 0.9567
    Epoch 2/20
    60000/60000 [==============================] - 6s 100us/step - loss: 0.1905 - acc: 0.9415 - val_loss: 0.1014 - val_acc: 0.9679
    Epoch 3/20
    60000/60000 [==============================] - 6s 102us/step - loss: 0.1485 - acc: 0.9538 - val_loss: 0.0830 - val_acc: 0.9736
    Epoch 4/20
    60000/60000 [==============================] - 6s 101us/step - loss: 0.1294 - acc: 0.9602 - val_loss: 0.0750 - val_acc: 0.9750
    Epoch 5/20
    60000/60000 [==============================] - 6s 99us/step - loss: 0.1126 - acc: 0.9642 - val_loss: 0.0717 - val_acc: 0.9767
    Epoch 6/20
    60000/60000 [==============================] - 6s 100us/step - loss: 0.1039 - acc: 0.9680 - val_loss: 0.0713 - val_acc: 0.9775
    Epoch 7/20
    60000/60000 [==============================] - 6s 102us/step - loss: 0.0940 - acc: 0.9708 - val_loss: 0.0692 - val_acc: 0.9791
    Epoch 8/20
    60000/60000 [==============================] - 6s 98us/step - loss: 0.0878 - acc: 0.9724 - val_loss: 0.0663 - val_acc: 0.9799
    Epoch 9/20
    60000/60000 [==============================] - 6s 98us/step - loss: 0.0822 - acc: 0.9731 - val_loss: 0.0659 - val_acc: 0.9783
    Epoch 10/20
    60000/60000 [==============================] - 6s 101us/step - loss: 0.0755 - acc: 0.9760 - val_loss: 0.0599 - val_acc: 0.9818
    Epoch 11/20
    60000/60000 [==============================] - 6s 102us/step - loss: 0.0753 - acc: 0.9754 - val_loss: 0.0617 - val_acc: 0.9798
    Epoch 12/20
    60000/60000 [==============================] - 6s 98us/step - loss: 0.0699 - acc: 0.9775 - val_loss: 0.0577 - val_acc: 0.9831
    Epoch 13/20
    60000/60000 [==============================] - 6s 101us/step - loss: 0.0680 - acc: 0.9774 - val_loss: 0.0587 - val_acc: 0.9824
    Epoch 14/20
    60000/60000 [==============================] - 6s 101us/step - loss: 0.0634 - acc: 0.9799 - val_loss: 0.0622 - val_acc: 0.9827
    Epoch 15/20
    60000/60000 [==============================] - 6s 103us/step - loss: 0.0588 - acc: 0.9809 - val_loss: 0.0573 - val_acc: 0.9826
    Epoch 16/20
    60000/60000 [==============================] - 6s 100us/step - loss: 0.0556 - acc: 0.9820 - val_loss: 0.0594 - val_acc: 0.9830
    Epoch 17/20
    60000/60000 [==============================] - 6s 102us/step - loss: 0.0552 - acc: 0.9814 - val_loss: 0.0539 - val_acc: 0.9844
    Epoch 18/20
    60000/60000 [==============================] - 6s 99us/step - loss: 0.0525 - acc: 0.9824 - val_loss: 0.0520 - val_acc: 0.9841
    Epoch 19/20
    60000/60000 [==============================] - 6s 99us/step - loss: 0.0508 - acc: 0.9831 - val_loss: 0.0565 - val_acc: 0.9827
    Epoch 20/20
    60000/60000 [==============================] - 6s 98us/step - loss: 0.0508 - acc: 0.9832 - val_loss: 0.0568 - val_acc: 0.9833
    


```python
#Test loss and Accuracy
score=model10.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 110us/step
    The test loss is  0.0567527698382
    The accuracy is  0.9833
    


```python
#Plotting the train and test error for each epochs
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAgAElEQVR4XuxdBbxcxfU+CQnuWhxaHIqXIgWCFoIVWrQUKe5FS9FQpIXi7gRLocX/JUigWIGW4u4EtxQnaJL/9+2+heXx5O7eOffOffPN73fyNvvuzJzznTO73xs5089UhIAQEAJCQAgIASEgBJJCoF9S1spYISAEhIAQEAJCQAgIARMBVBAIASEgBISAEBACQiAxBEQAE3O4zBUCQkAICAEhIASEgAigYkAICAEhIASEgBAQAokhIAKYmMNlrhAQAkJACAgBISAERAAVA0JACAgBISAEhIAQSAwBEcDEHC5zhYAQEAJCQAgIASEgAqgYEAJCQAgIASEgBIRAYgiIACbmcJkrBISAEBACQkAICAERQMWAEBACQkAICAEhIAQSQ0AEMDGHy1whIASEgBAQAkJACIgAKgaEgBAQAkJACAgBIZAYAiKAiTlc5goBISAEhIAQEAJCQARQMSAEhIAQEAJCQAgIgcQQEAFMzOEyVwgIASEgBISAEBACIoCKASEgBISAEBACQkAIJIaACGBiDpe5QkAICAEhIASEgBAQAVQMCAEhIASEgBAQAkIgMQREABNzuMwVAkJACAgBISAEhIAIoGJACAgBISAEhIAQEAKJISACmJjDZa4QEAJCQAgIASEgBEQAFQNCQAgIASEgBISAEEgMARHAxBwuc4WAEBACQkAICAEhIAKoGBACQkAICAEhIASEQGIIiAAm5nCZKwSEgBAQAkJACAgBEUDFgBAQAkJACAgBISAEEkNABDAxh8tcISAEhIAQEAJCQAiIACoGhIAQEAJCQAgIASGQGAIigIk5XOYKASEgBISAEBACQkAEUDEgBISAEBACQkAICIHEEBABTMzhMlcICAEhIASEgBAQAiKAigEhIASEgBAQAkJACCSGgAhgYg6XuUJACAgBISAEhIAQEAFUDAgBISAEhIAQEAJCIDEERAATc7jMFQJCQAgIASEgBISACKBiQAgIASEgBISAEBACiSEgApiYw2WuEBACQkAICAEhIAREABUDQkAICAEhIASEgBBIDAERwMQcLnOFgBAQAkJACAgBISACqBgQAkJACAgBISAEhEBiCIgAJuZwmSsEhIAQEAJCQAgIARFAxYAQEAJCQAgIASEgBBJDQAQwMYfLXCEgBISAEBACQkAIiAAqBoSAEBACQkAICAEhkBgCIoCJOVzmCgEhIASEgBAQAkJABFAxIASEgBAQAkJACAiBxBAQAUzM4TJXCAgBISAEhIAQEAIigIoBISAEhIAQEAJCQAgkhoAIYGIOl7lCQAgIASEgBISAEBABVAwIASEgBISAEBACQiAxBEQAE3O4zBUCQkAICAEhIASEgAigYkAICAEhIASEgBAQAokhIAKYmMNlrhAQAkJACAgBISAERAAVA0JACAgBISAEhIAQSAwBEcDEHC5zhYAQEAJCQAgIASEgAqgYEAJCQAgIASEgBIRAYgiIAOZzOPGbCfJxvmZUWwgIASEgBISAECgYgcnQ3xuQcQX3G0V3IoD53DAzqr+WrwnVFgJCQAgIASEgBEpCYBb0+3pJfZfarQhgPvgnR/UPX331VZt8cr7sm+Wrr76ym2++2VZffXUbOHBg3zSyw6qUbKXJKdkrW/vu0JVv+6ZvPf360Ucf2ayzzkrgpoB81DcR7NkqEcB8Xq8RQJQ+TwCHDx9ugwcPToIApmJrgwCmYi+/TGRrvg+8WGvLt7F6Jp9enn4lAZxiCnI/EcB8Xkq3tghgH/O95wdOjFClZK9sjTECw+gk34bBMbZWPP0qAmimGcB8ES8CmA+/6Gp7fuBEZ2zHErBmxWL0TD6dFMf58Iu5dkq+9bRVBFAEMO84FwHMi2Bk9T0/cCIztaZOSvbK1hgjMIxO8m0YHGNrxdOvIoAigHnjXQQwL4KR1ff8wInMVBHAGB0SSKcqxPG4cePs66+/tjFjxuS2mvbeeeedtsIKKySxT1m29h4y4403ng0YMMD69et6oVMEUASw9yjq+QkRwLwIRla/Cl+cISFLyV7ZGjJy8rX15Zdf2ptvvmmjR4/O11BHbZLJzz77zCaaaKJuv/CDdBRBI7I1uxMmnnhim3HGGW388cf/XiURQBHA7JHU9ZMigHkRjKx+SiRBS8CRBV9AdWKO47Fjx9pzzz1nnKGZbrrpal/O3c3SZIWEbX7yySc26aSTWv/+/bNWq+RzsrV3t5Ek84+Md999tzbDPPfcc38vLkQARQB7jyTNAGqfWN4oibh+zEQhNGyyNTSi7bX3+eef20svvWSzzz67cYYmRCEp4hc687GmQABla7ao4Qzzyy+/bHPOOadNOOGE36kkAigCmC2Kun9KM4B5EYysfkokQTOAkQVfQHVijuMGAezqS7ldCEQA20Uu7np5/dpTrIkAigDmjX4RwLwIRlY/5i9OD6hSsle2ekRQ622KALaOWXONvKQoX+/F1s5rqwhgz/5SHsB88SwCmA+/6GqnRBI0Axhd+AVTKOY4FgE023zzzY04XHHFFTWf/+xnP7Oll17ajj322G5jYJZZZrH999/fdt5551zL3Y12dt1112Dx5tWQCKAXsvV2RQDz4SsCmA+/6GrH/MXpAVZK9spWjwhqvc2qEsB11lmndtL4lltu+Z7R9957ry277LL2wAMP2OKLL94rKJ0J4HvvvVdLXzPZZJMFI4DnnntujTCOGjXqO23yYMQkk0wSbP9lVwoTo9VWW80+/vjj2sGcdosIYLvIZasnApgNp+6eEgHMh190tVMiCQQ/JXtlaxzDraoE8JprrrENNtjgmwMszWhut912dv/999tDDz2UCeTOBDBLpVZnALsjgFn6yvuMCGBeBIupLwKYD2cfAnjllWZXXWW2+upmW26ZT8MAtfXFGQDESJuQbyN1TE61YvZrVQkgk1aThO2000526KGHfuMhnjT9wQ9+YEcddZRxWZXY77DDDvbPf/7T3n77bZttttlq7++2227f1OltCfitt96ybbfd1m699dZaHju2vc8++3xnCfjss8+2iy66yF588UWbZpppbL311rOjjz66NrvXIGDNYXT44YfbQQcdVLOBM4ONJeCRI0fa7rvvXuuLiZPXXHNNO+WUU2opelhY58Ybb6zpf8ghh9iHH35oa621lp111lndzu71RgA5s3fYYYcZSSpnKBdccMGa7pw1ZPniiy/sd7/7nZF0v//++zV8ufS93377GVO8UI8LL7ywhu+0005rG220kZ1wwgnfGzXaA9jzB4kIYL4PWh8CiIFhQ4YYPgHMzjknn4YBasf8ZRLAvO80kZKtNDwle2Vr6NHSXntdfSnjOx1Jodtrj7XyLBUyE003l0V8TyESkL///e810tXIXUgiQsLHxNZTTTVVbW/fn//8Z1t77bVrxOxf//pX7feXXHJJbQaRpTcCuDr++H/nnXdqJItpbUjQHn74YfvLX/7yzR7A888/v7bcPMccc9gLL7xQI6ZrrLGGnXzyybUceKeeeqodeeSR9sQTT9T65PIyyWEzASRuiy66qE099dR2/PHH1+qxHerdWOomATzp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" width="640">


<h2> 11.MLP-ReLu-DP-BN-Adam (784-512-DP-BN-256-DP-BN-128-DP-BN-10)</h2> 


```python
#Initilaiisng the layer

model11=Sequential()

#Hidden Layer 1

model11.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model11.add(Activation('relu'))

#Batch Normalization Layer
model11.add(BatchNormalization())

#Drop out Layer
model11.add(Dropout(0.5))

#Hidden Layer 2
model11.add(Dense(256,kernel_initializer='he_normal'))
model11.add(Activation('relu'))

#Batch Normalization Layer
model11.add(BatchNormalization())

#Drop out Layer
model11.add(Dropout(0.5))

#Hidden Layer 3
model11.add(Dense(128,kernel_initializer='he_normal'))
model11.add(Activation('relu'))

#Batch Normalization Layer
model11.add(BatchNormalization())

#Drop out Layer
model11.add(Dropout(0.5))

#Output layer
model11.add(Dense(Output,kernel_initializer='glorot_normal'))
model11.add(Activation(tf.nn.softmax))

```


```python
#Model summary
model11.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_50 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    activation_50 (Activation)   (None, 512)               0         
    _________________________________________________________________
    batch_normalization_15 (Batc (None, 512)               2048      
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 512)               0         
    _________________________________________________________________
    dense_51 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    activation_51 (Activation)   (None, 256)               0         
    _________________________________________________________________
    batch_normalization_16 (Batc (None, 256)               1024      
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 256)               0         
    _________________________________________________________________
    dense_52 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    activation_52 (Activation)   (None, 128)               0         
    _________________________________________________________________
    batch_normalization_17 (Batc (None, 128)               512       
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 128)               0         
    _________________________________________________________________
    dense_53 (Dense)             (None, 10)                1290      
    _________________________________________________________________
    activation_53 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 571,018
    Trainable params: 569,226
    Non-trainable params: 1,792
    _________________________________________________________________
    


```python
#Compile
model11.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model11.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 11s 182us/step - loss: 0.5778 - acc: 0.8244 - val_loss: 0.1684 - val_acc: 0.9474
    Epoch 2/20
    60000/60000 [==============================] - 8s 127us/step - loss: 0.2529 - acc: 0.9261 - val_loss: 0.1144 - val_acc: 0.9668
    Epoch 3/20
    60000/60000 [==============================] - 7s 122us/step - loss: 0.1926 - acc: 0.9433 - val_loss: 0.1002 - val_acc: 0.9699
    Epoch 4/20
    60000/60000 [==============================] - 7s 123us/step - loss: 0.1645 - acc: 0.9510 - val_loss: 0.0896 - val_acc: 0.9729
    Epoch 5/20
    60000/60000 [==============================] - 8s 126us/step - loss: 0.1445 - acc: 0.9580 - val_loss: 0.0790 - val_acc: 0.9764
    Epoch 6/20
    60000/60000 [==============================] - 7s 125us/step - loss: 0.1304 - acc: 0.9619 - val_loss: 0.0807 - val_acc: 0.9761
    Epoch 7/20
    60000/60000 [==============================] - 8s 126us/step - loss: 0.1190 - acc: 0.9651 - val_loss: 0.0737 - val_acc: 0.9778
    Epoch 8/20
    60000/60000 [==============================] - 8s 127us/step - loss: 0.1102 - acc: 0.9671 - val_loss: 0.0715 - val_acc: 0.9796
    Epoch 9/20
    60000/60000 [==============================] - 8s 126us/step - loss: 0.1030 - acc: 0.9699 - val_loss: 0.0688 - val_acc: 0.9800
    Epoch 10/20
    60000/60000 [==============================] - 8s 126us/step - loss: 0.0952 - acc: 0.9715 - val_loss: 0.0608 - val_acc: 0.9813
    Epoch 11/20
    60000/60000 [==============================] - 8s 137us/step - loss: 0.0911 - acc: 0.9727 - val_loss: 0.0633 - val_acc: 0.9815
    Epoch 12/20
    60000/60000 [==============================] - 8s 126us/step - loss: 0.0891 - acc: 0.9731 - val_loss: 0.0685 - val_acc: 0.9803
    Epoch 13/20
    60000/60000 [==============================] - 8s 127us/step - loss: 0.0819 - acc: 0.9754 - val_loss: 0.0606 - val_acc: 0.9813
    Epoch 14/20
    60000/60000 [==============================] - 8s 128us/step - loss: 0.0801 - acc: 0.9764 - val_loss: 0.0635 - val_acc: 0.9815
    Epoch 15/20
    60000/60000 [==============================] - 8s 125us/step - loss: 0.0703 - acc: 0.9793 - val_loss: 0.0620 - val_acc: 0.9827
    Epoch 16/20
    60000/60000 [==============================] - 8s 126us/step - loss: 0.0720 - acc: 0.9785 - val_loss: 0.0616 - val_acc: 0.9837
    Epoch 17/20
    60000/60000 [==============================] - 8s 127us/step - loss: 0.0716 - acc: 0.9780 - val_loss: 0.0617 - val_acc: 0.9819
    Epoch 18/20
    60000/60000 [==============================] - 8s 128us/step - loss: 0.0686 - acc: 0.9796 - val_loss: 0.0591 - val_acc: 0.9839
    Epoch 19/20
    60000/60000 [==============================] - 7s 124us/step - loss: 0.0648 - acc: 0.9805 - val_loss: 0.0597 - val_acc: 0.9823
    Epoch 20/20
    60000/60000 [==============================] - 7s 124us/step - loss: 0.0611 - acc: 0.9809 - val_loss: 0.0610 - val_acc: 0.9830
    


```python
#Test loss and Accuracy
score=model11.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The Accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 138us/step
    The test loss is  0.0609957450023
    The Accuracy is  0.983
    


```python
#Plotting the train and test error for each epochs
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAgAElEQVR4XuxdBZhdRdKtGIQQ3F0Wd/vRxQkSHBYLFhZ3WWBxgiyL2y7ulsWdIMHd3TW4BQsQJCT567w3L7xMRu59t6u77+vT31fMMLm3u+pU1ZszLdVdhI0IEAEiQASIABEgAkQgKQS6JGUtjSUCRIAIEAEiQASIABEQEkAGAREgAkSACBABIkAEEkOABDAxh9NcIkAEiAARIAJEgAiQADIGiAARIAJEgAgQASKQGAIkgIk5nOYSASJABIgAESACRIAEkDFABIgAESACRIAIEIHEECABTMzhNJcIEAEiQASIABEgAiSAjAEiQASIABEgAkSACCSGAAlgYg6nuUSACBABIkAEiAARIAFkDBABIkAEiAARIAJEIDEESAATczjNJQJEgAgQASJABIgACSBjgAgQASJABIgAESACiSFAApiYw2kuESACRIAIEAEiQARIABkDRIAIEAEiQASIABFIDAESwMQcTnOJABEgAkSACBABIkACyBggAkSACBABIkAEiEBiCJAAJuZwmksEiAARIAJEgAgQARJAxgARIAJEgAgQASJABBJDgAQwMYfTXCJABIgAESACRIAIkAAyBogAESACRIAIEAEikBgCJICJOZzmEgEiQASIABEgAkSABJAxQASIABEgAkSACBCBxBAgAUzM4TSXCBABIkAEiAARIAIkgIwBIkAEiAARIAJEgAgkhgAJYGIOp7lEgAgQASJABIgAESABZAwQASJABIgAESACRCAxBEgAE3M4zSUCRIAIEAEiQASIAAkgY4AIEAEiQASIABEgAokhQAKYmMNpLhEgAkSACBABIkAESAAZA0SACBABIkAEiAARSAwBEsDEHE5ziQARIAJEgAgQASJAAsgYIAJEgAgQASJABIhAYgiQACbmcJpLBIgAESACRIAIEAESQMYAESACRIAIEAEiQAQSQ4AEMDGH01wiQASIABEgAkSACJAAMgaIABEgAkSACBABIpAYAiSAiTmc5hIBIkAEiAARIAJEgASQMUAEiAARIAJEgAgQgcQQIAFMzOE0lwgQASJABIgAESACJICMASJABIgAESACRIAIJIYACWBiDqe5RIAIEAEiQASIABEgAWQMEAEiQASIABEgAkQgMQRIABNzOM0lAkSACBABIkAEiAAJIGOACBABIkAEiAARIAKJIUACmJjDaS4RIAJEgAgQASJABEgAGQNEgAgQASJABIgAEUgMARLAxBxOc4kAESACRIAIEAEiQALIGCACRIAIEAEiQASIQGIIkAAm5nCaSwSIABEgAkSACBABEkDGABEgAkSACBABIkAEEkOABDAxh9NcIkAEiAARIAJEgAiQADIGiAARIAJEgAgQASKQGAIkgIk5nOYSASJABIgAESACRIAEkDFABIgAESACRIAIEIHEECABTMzhNJcIEAEiQASIABEgAs1GAHdTlx6gMp3Kayr7qDzSgZsn1X/7l8pGKpOpfKDyD5VBDA0iQASIABEgAkSACDQrAs1EADdTJ12hAhL4mMrOKjuozKfyURsOHK/lua/063Eqn6jMpPKjyksZHQ78pm95J+MrfIwIEAEiQASIABGIAIGJVIfPVEZHoIt3FZqJAD6l6D2vsmsdim/o9zerHNwGsrvozzBbOI/KiAaRn6GFODb4Ol8jAkSACBABIkAEAiIwo479acDxgw3dLAQQs3nDVTZRuakOzTP0+0VUVmwDYSzzftvy3vr69WuVgSonqIxsxyPj688htYa/Hj754IMPZKKJ8G1zthEjRsgDDzwgK6+8svTo0aM5jWyxKiVbYXJK9tLW5k1d+rY5fWvp1x9//FFmm202ADeJyrDmRLBjq5qFAGIZFgx+OZXH60w+RL/fVmXuNmB4U382q8pVKmerzKlylgpI49HtwDZAf35k638bOHCg9OrVK8X4oc1EgAgQASJABEqHwPDhw6Vfv34kgKXz3LgK1wjgsvpPT9T986H6/dYqWOZt3d7WH/RUwZ8AtRm//fT72iGStmBpcwZw6NChMvHEEzcBjG2bgL/CBg8eLH369EliBjAVW+Ft+rY50zYlvzKOmzOGrf06bNgwmXLKKUkAmyB8GlkCfgi//1RWq7N/Lf0eS8Mger9nwAWs7wdtTU8ABw0aJH379k2CAKZia+0DNhV7QYpoa4ZPtRI+Qt+W0GkZVLb0KwjgJJNg9ZdLwBlcEf0jOATynApOAdfa6/rNLSptHQLByV/M/86uMqrlhb316z9VMKOYpZEAZkGpRM9YfuDECENK9tLWGCPQjU70rRscY+vF0q8kgCLNsgcQcVsrA4PTvVgG3kllR5X5VT5UuVwF+wRrZBAlX0AQL1X5jwr2AF6scqYKagNmaSSAWVAq0TOWHzgxwpCSvbQ1rggcPXq0/PHHHzJyZHtn7rLrC98+/PDDssIKKySxSkFbO4+Nbt26Sffu3aVLl7ZpDglgcxFARARm/w5UQSHoV1X2VXm4JVQe1K9DVPrXhc4y+v1pKjgpDHJ4kUpHp4BbRx0JYOd5WKonUiIJcExK9tLWeFLx999/l88//1ywEd9FA5n85ZdfZIIJJmj3F76LcWLog7Zm9wIOZ0433XQy3njYJTZ2IwFsPgKYPTLcPEkC6AbHaHpJiSSQAEYTds4ViTmOR40aJe+8845ghmaqqaaq/HJub5YmKzDo86effpLevXtL165ds75Wyudoa+duA0nGHxlff/11ZYZ5zjnnHCcuSABJADuPpI6fIAEsimBk78f8i9MCqpTspa0WEZS/z19//VVQO3WWWWZxVj4LpAi/0FGNIQUCSFuzxR1mmD/88MNKvb+ePVH0489GAkgCmC2K2n+KBLAogpG9nxJJ4AxgZMHnUJ2Y47hGANv6pdwoBCSAjSIX93tF/dpRrJEAkgAWjX4SwKIIRvZ+zL84LaBKyV7aahFB+fskAcyPWf0bRUlRsdH9vl3UVhLAjv3VTKeA/UZmdTQSwBCoG46ZEkngDKBhIAXuOuY4JgEU2WqrrQQ4XH/99ZVI+etf/ypLL720nHzyye1GzowzzigHHXSQ7LbbboWWu2v97LHHHoGjtPPhSQA7x6jIEySARdAjASyGXoRvx/yL0wKulOylrRYRlL/PshLAddddt3LS+N577x3H6CeeeEKWXXZZee6552SxxRbrFJTWBPDbb7+tlK/p6E75vATwwgsvrBBG3FRV33AwYsIJJ3S2/7ItY4ERbo7Cfbs4mNNoIwFsFLls75EAZsOpvac4A1gMv+jeTokkcAYwuvBzplDMcVxWAnjzzTfLRhttNOYAS72zdtxxR3n22WflhRdeyOTD1gQwy0uuCGCWsYo+QwJYFEE/75MAFsPZhgDecIPIjTeKrL66yLbbFtPQwdsx/zJxYN5YXaRkKwmg6+iJp7+Y47isBBBFq0HCdt11VznyyCPHOBsnTaeddlo57rjjBMuqwH7nnXeW+++/X7788kuZeeaZKz/fc889x7zT2RLwF198ITvssIPcd999lTp26Hv//fcfawn4/PPPl8svv1zef/99mWKKKWT99deXE044oTK7VyNg9RF5zDHHyGGHHVaxATODtSXgIUOGyF577VUZC4WT11prLfnPf/5TKdGDhnfuuuuuiv5HHHEE7j6VtddeW84777x2Z/c6I4CY2TvqqKMEs5SYoZx//vkrumPWEO23336TffbZR0C6v/vuuwq+WPo+8MADBSVeoMdll11WwRf3+W666aZy2mko6Tt24x7Ajj+TSACLfWbbEEBNDBkwQPQTQOSCC4pp6ODtmH+ZODCPBJD3PLsOo+D9xZyzbf1S1t/pWhS6cdiKLBVqrWCtQ5htbBCQ6667rkK6arULQURA+FDYerLJJqvs7Tv++ONlnXXWqRCzRx99tPLvV155ZWUGEa0zAri6/vH/1VdfVUgWytqAoL344oty0kknjdkDePHFF1eWm2eddVZ57733KsR0zTXXlDPPPLNSA++///2v/Otf/5LXXnutMiaWl0EO6wkgcFtkkUVk8sknl1NPPbXyHvqB3rWlbhDAM844o0IMQby++eabCuHaZZddKiSurdYZAYQd0A0kduGFF9ZfcxdUSOcbb7whs88+ewW/c889t0JwJ5100goJBL6bb765XH311ZWxr7nmGpl33nkrP3/11Vdl++23JwHMFsZjnsoY9jl7TedxGwKoia3RLLLGGqJ/egVHM+ZfJq7BSclWYJeSvbTVdbY01l9bBPDnn0Vnkxrrr+hbWj9aiVG2Xt58880K6cDs3sorr1x5acUVV5QZZphBBg4c2G4nIICYOQN56YwAvv7665UZMSwpL7744pXnQXAWXHDBCklq7xDI//73P9l3330Fs4do7e0BrCeAd955p6y33nqCWUDYgPbyyy9XSNnzzz8viy66aGUGEAQQ/YJAou23337y9NNPV8htW60zAjjNNNPIP/7xj8qMXq2BzC6//PKVsWDju+++K9AP+wjr6zueeOKJcumll1b0xIxlR40zgB3HNQlgtrxv7ykbAjh4cHX5d775RP98K6ahg7f5i9MBiJF2Qd9G6piCasXs1zITQLhlueWWq8xSXXHFFZWZN9wycc8998hqq602xmtnn322YIYORYhxcAQza0sssYQ8/vjjnRLAG3QLUL9+/SozifU3pIAEYSm4RgBBwLBsClIKcokbL/AOZPzxx89EADHrd84551RuZqlvmC3E7CP0AAG87bbb5KWXXhrzCGbwMGv39ttv5yaAOPBSmxkF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" width="640">


<h2> 12. MLP-ReLu-BN-DP-Adam(784-512-BN-DP-256-BN-DP-128-BN-DP-64-BN-DP-32-BN-DP-10)</h2>


```python
#Initialising all layers
model12=Sequential()

# Hidden Layer 1
model12.add(Dense(512,input_dim=Input,kernel_initializer='he_normal'))
model12.add(Activation('relu'))

#Batch Normalization Layer
model12.add(BatchNormalization())

#Dropout Layer
model12.add(Dropout(0.5))

#Hidden layer 2
model12.add(Dense(256,kernel_initializer='he_normal'))
model12.add(Activation('relu'))

#Batch Normalization Layer
model12.add(BatchNormalization())

#Dropout Layer
model12.add(Dropout(0.5))

#Hidden Layer 3
model12.add(Dense(128,kernel_initializer='he_normal'))
model12.add(Activation('relu'))

#Batch Normalization Layer
model12.add(BatchNormalization())

#Dropout Layer
model12.add(Dropout(0.5))

#Hidden layer 4
model12.add(Dense(64,kernel_initializer='he_normal'))
model12.add(Activation('relu'))

#Batch Normalization Layer
model12.add(BatchNormalization())

#Dropout Layer
model12.add(Dropout(0.5))

#Hidden Layer 5
model12.add(Dense(32,kernel_initializer='he_normal'))
model12.add(Activation('relu'))

#Batch Normalization Layer
model12.add(BatchNormalization())

#Dropout Layer
model12.add(Dropout(0.5))

#Output Layer
model12.add(Dense(Output,kernel_initializer='glorot_normal'))
model12.add(Activation(tf.nn.softmax))

```


```python
#Model Summary
model12.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_54 (Dense)             (None, 512)               401920    
    _________________________________________________________________
    activation_54 (Activation)   (None, 512)               0         
    _________________________________________________________________
    batch_normalization_18 (Batc (None, 512)               2048      
    _________________________________________________________________
    dropout_16 (Dropout)         (None, 512)               0         
    _________________________________________________________________
    dense_55 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    activation_55 (Activation)   (None, 256)               0         
    _________________________________________________________________
    batch_normalization_19 (Batc (None, 256)               1024      
    _________________________________________________________________
    dropout_17 (Dropout)         (None, 256)               0         
    _________________________________________________________________
    dense_56 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    activation_56 (Activation)   (None, 128)               0         
    _________________________________________________________________
    batch_normalization_20 (Batc (None, 128)               512       
    _________________________________________________________________
    dropout_18 (Dropout)         (None, 128)               0         
    _________________________________________________________________
    dense_57 (Dense)             (None, 64)                8256      
    _________________________________________________________________
    activation_57 (Activation)   (None, 64)                0         
    _________________________________________________________________
    batch_normalization_21 (Batc (None, 64)                256       
    _________________________________________________________________
    dropout_19 (Dropout)         (None, 64)                0         
    _________________________________________________________________
    dense_58 (Dense)             (None, 32)                2080      
    _________________________________________________________________
    activation_58 (Activation)   (None, 32)                0         
    _________________________________________________________________
    batch_normalization_22 (Batc (None, 32)                128       
    _________________________________________________________________
    dropout_20 (Dropout)         (None, 32)                0         
    _________________________________________________________________
    dense_59 (Dense)             (None, 10)                330       
    _________________________________________________________________
    activation_59 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 580,778
    Trainable params: 578,794
    Non-trainable params: 1,984
    _________________________________________________________________
    


```python
#Compile
model12.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model12.fit(X_train,Y_train,batch_size=batch,epochs=epochs,validation_data=(X_test,Y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 15s 245us/step - loss: 1.4905 - acc: 0.5009 - val_loss: 0.4474 - val_acc: 0.9027
    Epoch 2/20
    60000/60000 [==============================] - 10s 174us/step - loss: 0.6451 - acc: 0.8110 - val_loss: 0.2092 - val_acc: 0.9404
    Epoch 3/20
    60000/60000 [==============================] - 10s 173us/step - loss: 0.4205 - acc: 0.8910 - val_loss: 0.1544 - val_acc: 0.9568
    Epoch 4/20
    60000/60000 [==============================] - 10s 173us/step - loss: 0.3317 - acc: 0.9176 - val_loss: 0.1383 - val_acc: 0.9634
    Epoch 5/20
    60000/60000 [==============================] - 11s 186us/step - loss: 0.2789 - acc: 0.9333 - val_loss: 0.1257 - val_acc: 0.9696
    Epoch 6/20
    60000/60000 [==============================] - 10s 172us/step - loss: 0.2562 - acc: 0.9399 - val_loss: 0.1234 - val_acc: 0.9698
    Epoch 7/20
    60000/60000 [==============================] - 11s 175us/step - loss: 0.2362 - acc: 0.9464 - val_loss: 0.1078 - val_acc: 0.9732
    Epoch 8/20
    60000/60000 [==============================] - 10s 172us/step - loss: 0.2153 - acc: 0.9523 - val_loss: 0.1113 - val_acc: 0.9739
    Epoch 9/20
    60000/60000 [==============================] - 10s 169us/step - loss: 0.2065 - acc: 0.9524 - val_loss: 0.0999 - val_acc: 0.9767
    Epoch 10/20
    60000/60000 [==============================] - 10s 172us/step - loss: 0.1913 - acc: 0.9566 - val_loss: 0.0998 - val_acc: 0.9760
    Epoch 11/20
    60000/60000 [==============================] - 10s 175us/step - loss: 0.1793 - acc: 0.9593 - val_loss: 0.1026 - val_acc: 0.9744
    Epoch 12/20
    60000/60000 [==============================] - 11s 179us/step - loss: 0.1765 - acc: 0.9593 - val_loss: 0.0913 - val_acc: 0.9785
    Epoch 13/20
    60000/60000 [==============================] - 10s 171us/step - loss: 0.1674 - acc: 0.9628 - val_loss: 0.0969 - val_acc: 0.9777
    Epoch 14/20
    60000/60000 [==============================] - 10s 173us/step - loss: 0.1573 - acc: 0.9650 - val_loss: 0.0949 - val_acc: 0.9788
    Epoch 15/20
    60000/60000 [==============================] - 10s 171us/step - loss: 0.1540 - acc: 0.9651 - val_loss: 0.0887 - val_acc: 0.9787
    Epoch 16/20
    60000/60000 [==============================] - 10s 173us/step - loss: 0.1458 - acc: 0.9671 - val_loss: 0.0851 - val_acc: 0.9804
    Epoch 17/20
    60000/60000 [==============================] - 10s 171us/step - loss: 0.1428 - acc: 0.9674 - val_loss: 0.0871 - val_acc: 0.9796
    Epoch 18/20
    60000/60000 [==============================] - 10s 172us/step - loss: 0.1426 - acc: 0.9678 - val_loss: 0.0804 - val_acc: 0.9817
    Epoch 19/20
    60000/60000 [==============================] - 10s 174us/step - loss: 0.1348 - acc: 0.9699 - val_loss: 0.0834 - val_acc: 0.9798
    Epoch 20/20
    60000/60000 [==============================] - 10s 172us/step - loss: 0.1308 - acc: 0.9709 - val_loss: 0.0801 - val_acc: 0.9813
    


```python
#Test loss and accuracy
score=model12.evaluate(X_test,Y_test)
print("The test loss is ",score[0])
print("The accuracy is ",score[1])
```

    10000/10000 [==============================] - 1s 138us/step
    The test loss is  0.0800626712134
    The accuracy is  0.9813
    


```python
#Plotting the train and test error for each epochs
fig,ax = plt.subplots(1,1)
ax.set_xlabel('Epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')

# list of epoch numbers
x = list(range(1,epochs+1))

# print(history.history.keys())
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# we will get val_loss and val_acc only when you pass the paramter validation_data
# val_loss : validation loss
# val_acc : validation accuracy

# loss : training loss
# acc : train accuracy
# for each key in histrory.histrory we will have a list of length equal to number of epochs

vy = his.history['val_loss']
ty = his.history['loss']
plt_dynamic(x, vy, ty, ax)
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAgAElEQVR4XuxdB5iURdIuECQoioo5YQ6Y9bwzY8CAp55ZOfNvFuOZ9RRzwJzPM3tiOrOCijnnnCMKZsyKifDXO7PDDcvuzjfzdXX3t/3289TDAh3qraqeebdDdQdhoQVoAVqAFqAFaAFagBZIygIdkkJLsLQALUAL0AK0AC1AC9ACQgLIIKAFaAFagBagBWgBWiAxC5AAJuZwwqUFaAFagBagBWgBWoAEkDFAC9ACtAAtQAvQArRAYhYgAUzM4YRLC9ACtAAtQAvQArQACSBjgBagBWgBWoAWoAVogcQsQAKYmMMJlxagBWgBWoAWoAVoARJAxgAtQAvQArQALUAL0AKJWYAEMDGHEy4tQAvQArQALUAL0AIkgIwBWoAWoAVoAVqAFqAFErMACWBiDidcWoAWoAVoAVqAFqAFSAAZA7QALUAL0AK0AC1ACyRmARLAxBxOuLQALUAL0AK0AC1AC5AAMgZoAVqAFqAFaAFagBZIzAIkgIk5nHBpAVqAFqAFaAFagBYgAWQM0AK0AC1AC9ACtAAtkJgFSAATczjh0gK0AC1AC9ACtAAtQALIGKAFaAFagBagBWgBWiAxC5AAJuZwwqUFaAFagBagBWgBWoAEkDFAC9ACtAAtQAvQArRAYhYgAUzM4YRLC9ACtAAtQAvQArQACSBjgBagBWgBWoAWoAVogcQsQAKYmMMJlxagBWgBWoAWoAVoARJAxgAtQAvQArQALUAL0AKJWYAEMDGHEy4tQAvQArQALUAL0AIkgIwBWoAWoAVoAVqAFqAFErMACWBiDidcWoAWoAVoAVqAFqAFSAAZA7QALUAL0AK0AC1ACyRmARLAxBxOuLQALUAL0AK0AC1AC5AAMgZoAVqAFqAFaAFagBZIzAIkgIk5nHBpAVqAFqAFaAFagBYgAWQM0AK0AC1AC9ACtAAtkJgFSAATczjh0gK0AC1AC9ACtAAtQALIGKAFaAFagBagBWgBWiAxC5AAJuZwwqUFaAFagBagBWgBWoAEkDFAC9ACtAAtQAvQArRAYhYgAUzM4YRLC9ACtAAtQAvQArQACSBjgBagBWgBWoAWoAVogcQsQAKYmMMJlxagBWgBWoAWoAVoARJAxgAtQAvQArQALUAL0AKJWYAEMDGHEy4tQAvQArQALUAL0AIkgIwBWoAWoAVoAVqAFqAFErMACWBiDidcWoAWoAVoAVqAFqAFSAAZA7QALUAL0AK0AC1ACyRmARLAxBxOuLQALUAL0AK0AC1AC5AAMgZoAVqAFqAFaAFagBZIzAIkgIk5nHBpAVqAFqAFaAFagBYgAWQM0AK0AC1AC9ACtAAtkJgFSAATczjh0gK0AC1AC9ACtAAtQALIGKAFaAFagBagBWgBWiAxC5AAJuZwwqUFaAFagBagBWgBWoAEkDFAC9ACtAAtQAvQArRAYhYgAUzM4YRLC9ACtAAtQAvQArQACSBjgBagBWgBWoAWoAVogcQsQAKYmMMJlxagBWgBWoAWoAVoARJAxgAtQAvQArQALUAL0AKJWYAEMJ/DYb/ZVH7M1w1b0wK0AC1AC9ACtIBnC/TQ8T5VmeB53CiGIwHM54bZtfmofF2wNS1AC9ACtAAtQAsEssAcOu4ngcYOOiwJYD7zT6PNvx85cqRMMw1+bJ/ljz/+kHvvvVfWXntt6dy5c/sE2YQqJayAnBJeYm2/U5e+bZ++tfTrDz/8IHPOOScMN63KD+3Tgm2jIgHM5/USAdTS7gng0KFDpX///kkQwFSwVghgKnjxZUKs+T7wYm1N38bqmXx6WfoVBHDaacH9SADzeSnd1iSA7cz3lh84MZoqJbzEGmMEutGJvnVjx9h6sfQrCaAIVwDzRTwJYD77Rdfa8gMnOrBNW8BcFYvRM/l0Yhzns1/MrVPyrSVWEkASwLzznAQwrwUja2/5gRMZ1JI6KeEl1hgj0I1O9K0bO8bWi6VfSQBJAPPGOwlgXgtG1t7yAycyqCSAMTrEkU5FiOMJEybI2LFjZdy4cblRA+8jjzwiq666ahLnlIm1dshMMcUU0qlTJ+nQoeWNThJAEsDaUdR2DRLAvBaMrH0RvjhdmiwlvMTqMnLy9fX777/LZ599JmPGjMnXUVNrkMlffvlFunXr1uoXvpOBIuiEWLM7oXv37jLrrLPKlFNOOVkjEkASwOyR1HJNEsC8FoysfUokgVvAkQWfQ3VijuPx48fLu+++K1ihmXHGGUtfzq2t0mQ1Cfr86aefZOqpp5aOHTtmbVbIesRa220gyfgl46uvviqtMC+wwAKTxQUJIAlg7UjiCiDPieWNkojbx0wUXJuNWF1btLH+fv31V/nwww9l7rnnFqzQuCggRfhCRz7WFAggsWaLGqwwf/TRRzLPPPNI165dJ2lEAkgCmC2KWq/FFcC8FoysfUokgSuAkQWfQ3VijuMKAWzpS7lRE5AANmq5uNvl9WtbsUYCSAKYN/pJAPNaMLL2MX9xWpgqJbzEahFB9fdJAli/zapb5CVF+Ub32zovVhLAtv3FPID54pkEMJ/9omudEkngCmB04edMoZjjmARQZJttthHY4b///W/J5yuvvLL85S9/kdNOO63VGJhjjjnk0EMPlT333DPXdneln4EDBzqLN6uOSACtLFvulwQwn31JAPPZL7rWMX9xWhgrJbzEahFB9fdZVAK4wQYblG4a33fffZOBfvLJJ2XFFVeU559/XpZZZpmaRmlOAL/55ptS+poePXo4I4CXXHJJiTCOHj16kj5xMWKqqaZydv6yJYVho379+smPP/5YupjTaCEBbNRy2dqRAGazU2u1SADz2S+61imRBBg/JbzEGsd0KyoBvPXWW2WTTTaZeIGl2pq77LKLPPfcc/Liiy9mMnJzApilUb0rgK0RwCxj5a1DApjXgn7akwDms7MNAbz5ZtG9AZF11hHZfvt8GjpozS9OB0aMtAv6NlLH5FQrZr8WlQAiaTVI2B577CFHH330RA/hpukss8wiJ554omBbFbbfbbfd5IEHHpAvvvhC5pprrtK/77333hPb1NoC/vzzz2XnnXeW+++/v5THDn0feOCBk2wBX3zxxXLVVVfJBx98IDPMMINstNFGcsopp5RW9yoErDqMjjvuODnyyCNLGLAyWNkCHjFihOyzzz6lsZA4eb311pNzzz23lKIHBW3uvvvukv5HHXWUfP/997L++uvLv/71r1ZX92oRQKzsHXPMMQKSihXKPn36lHTHqiHKb7/9Jvvtt5+AdH/77bcl+2Lr++CDDxakeIEeV155Zcm+vXr1ki222ELOPPPMyWYNzwC2/UFCApjvg9aGAB57rOgnjOgngMi//51PQwetY/4ycQBvki5SwgrgKeElVtezpbH+WvpS1u90TQrdWH9olWerEJloWnksYjKFQEBuvPHGEumq5C4EEQHhQ2Lr6aabrnS27+STT5a//vWvJWL22GOPlf7/P//5T2kFEaUWAVx77bXlyy+/LJEspLUBQXvppZdk8ODBE88AXnbZZaXt5t69e8v7779fIqbrrruunHPOOaUceOedd56ccMIJ8vrrr5fGxPYyyGE1AYTdllpqKZl++unljDPOKLVDP9C7stUNAnj22WeXiCGI19dff10iXLvvvnuJxLVUahFA4IBuILFLLrmkfs39u0Q633zzTZl33nlL9rvoootKBLdnz54lEgj7brXVVnLdddeVxr7++utlkUUWKf37a6+9Jv/3f/9HAljnFCIBrNNgzarbEMArrhDZcUcR/RCQe+7Jp6GD1vzidGDESLugbyN1TE61YvZrSwTw559FV5Nygm6wueaPVmKUrfFbb71VIh1Y3Vt99dVLjVZbbTWZffbZZciQIa12AgKIlTOQl1oE8I033iitiGFLedllly3VB8FZfPHFSySptUsg1157rey///6C1UOU1raAqwngsGHDZMMNNxSsAgIDyiuvvFIiZS+88IIsvfTSpRVAEED0CwKJcsABB8gzzzxTIrctlVoEcOaZZ5Z//OMfpRW9SgGZXWWVVUpjAeN7770n0A/nCKvzO5566qlyhX5HQk+sWLZVuALYdlyTAGab963VsiGA+uEia64psvDCor8S5dPQQeuYv0wcwJuki5SwAnhKeInV9WxprL8iE0AgXmmllUqrVFdffXVp5Q2vTNx7772y1lprTTTIBRdcIFihQxJiXBzBytpyyy0nTzzxRKlOWyuAN910kwwYMKC0klj9QgpIELaCKwQQBAzbpiClIJd48QJtIF26dMlEALHqd+GFF5ZeZqkuWC3E6iP0AAG844475OWXX55YBSt4WLV755136iaAuPBSWRmFLSsFW8xvv/12yZbPPvusrn+sLTPNNFOJaGPlFH9HgU3RDiujWPHs37+/4IIOXpZpXkgASQAb+5TK1sqGAOpvPvqpInpNS/R9o+z7E9l0rrsWvzjrNllhGtC3hXFVXYrG7NcibwHDCSB2OD+HFTGsRl1zzTWTbAljJRDn90Cu/vznP5e2XrGliS1crOrVIoBIDQOCCOJYTQDRz0knnVQigK+++mqp77322qu0HYut54cfflh23XXXiTdvs6wAnn766SWi15zIYSxsz2699dYTzwBWdIf+SFeDLVqs0rVU2loBxBYyzu09/vjjpZvTlQKbgoje07TrBVJ71113lc4f3nbbbaUt6MoKKmwDoohxsCUPEv7ggw9OtiJIAkgCWNcHZ52VbQigHoDVd2vKquAKv57HCFli/jJxbZeUsMJ2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" width="640">


<h2><u> Conclusions</u> </h2>

<h3> MLP Without Batch Normalization and Dropouts </h3>

<table style="width:100%">
  <tr>
    <th>Layers</th>  
    <th>Training Accuracy</th>
    <th>Test Accuracy </th> 
    <th>Train Loss</th>
    <th>Test loss</th>
  </tr>
  <tr>
    <td>784-512-256-10 </td>  
    <td>99.64</td>
    <td>97.50</td> 
    <td>0.0123</td>
    <td>0.1195</td> 
  </tr>
  <tr>
    <td>784-512-256-128-10 </td>  
    <td>99.76</td>
    <td>98.25</td> 
    <td>0.0076</td>
    <td>0.0839</td> 
  </tr>
  <tr>
    <td>784-512-256-128-64-32-10 </td>  
    <td>99.65</td>
    <td>98.42</td> 
    <td>0.011</td>
    <td>0.0756</td> 
  </tr>
</table>

<h3> MLP With Batch Normalization </h3>

<table style="width:100%">
  <tr>
    <th>Layers</th>  
    <th>Training Accuracy</th>
    <th>Test Accuracy </th> 
    <th>Train Loss</th>
    <th>Test loss</th>
  </tr>
  <tr>
    <td>784-512-BN-256-BN-10 </td>  
    <td>99.64</td>
    <td>97.50</td> 
    <td>0.0123</td>
    <td>0.1195</td> 
  </tr>
  <tr>
    <td>784-512-BN-256-BN-128-BN-10 </td>  
    <td>99.76</td>
    <td>98.25</td> 
    <td>0.0076</td>
    <td>0.0839</td> 
  </tr>
  <tr>
    <td>784-512-BN-256-BN-128-BN-64-BN-32-BN-10 </td>  
    <td>99.65</td>
    <td>98.42</td> 
    <td>0.011</td>
    <td>0.0756</td> 
  </tr>
</table>

<h3> MLP With Drop outs </h3>

<table style="width:100%">
  <tr>
    <th>Layers</th>  
    <th>Training Accuracy</th>
    <th>Test Accuracy </th> 
    <th>Train Loss</th>
    <th>Test loss</th>
  </tr>
  <tr>
    <td>784-512-DP-256-DP-10 </td>  
    <td>98.44</td>
    <td>98.30</td> 
    <td>0.0510</td>
    <td>0.0617</td> 
  </tr>
  <tr>
    <td>784-512-DP-256-DP-128-DP-10 </td>  
    <td>98.21`</td>
    <td>98.24</td> 
    <td>0.0608</td>
    <td>0.0644</td> 
  </tr>
  <tr>
    <td>784-512-DP-256-DP-128-DP-64-DP-32-DP-10 </td>  
    <td>95.63</td>
    <td>97.44</td> 
    <td>0.1851</td>
    <td>0.1242</td> 
  </tr>
</table>

<h3> MLP With Batch Normalization and Dropouts </h3>

<table style="width:100%">
  <tr>
    <th>Layers</th>  
    <th>Training Accuracy</th>
    <th>Test Accuracy </th> 
    <th>Train Loss</th>
    <th>Test loss</th>
  </tr>
  <tr>
    <td>784-512-BN-DP-256-BN-DP-10 </td>  
    <td>98.32</td>
    <td>98.33</td> 
    <td>0.0508</td>
    <td>0.0568</td> 
  </tr>
  <tr>
    <td>784-512-BN-DP-256-BN-DP-128-BN-DP-10 </td>  
    <td>98.09</td>
    <td>98.30</td> 
    <td>0.0611</td>
    <td>0.0610</td> 
  </tr>
  <tr>
    <td>784-512-BN-DP-256-BN-DP-128-BN-DP-64-BN-DP-32-BN-DP-10 </td>  
    <td>97.09</td>
    <td>98.13</td> 
    <td>0.1308</td>
    <td>0.0801</td> 
  </tr>
</table>
