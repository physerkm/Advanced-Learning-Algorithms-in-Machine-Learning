# **Machine Learning Specialization**

## **Advanced Learning Algorithms**

### **TensorFlow implementation**

This week, we're going to go over training of a neural network. I think being able to take your own data and train your own neural network on it is really fun. We'll look at how you could do that.

#### **Train a Neural Network in TensorFlow**

Let's continue with our running example of handwritten digit recognition recognizing this image as zero or a one. Here we're using the neural network architecture, where you have an input X, that is the image, and then the first hidden layer was 25 units, second hidden layer with 15 units, and then one output unit. If you're given a training set of examples comprising images X, as was the ground truth label Y, how would you train the parameters of this neural network?


**First Step**
```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
  model = Sequential ({
    Dense(units=25, activation='sigmoid')
    Dense(units=15, activation='sigmoid')
    Dense(units=1, activation='sigmoid'))]
```


The first hidden layer with 25 units and sigmoid activation, the second hidden layer, and then finally the output layer.


**Second Step**
```
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss=BinaryCrossentropy())
```

Second step is you're to ask TensorFlow to compile the model. The key step in asking TensorFlow to compile the model is to specify what is the loss function you want to use. In this case we'll use something that goes by the Arcane name of sparse categorical "crossentropy".

**Third Step**
```
model.fit(X, Y, epochs=100)
```

Then having specified the loss function, the third step is to call the fit function, which tells TensorFlow to fit the model that you specified in step 1 using the loss of the cost function that you specified in step 2 to the dataset X, Y.

Back to the gradient descent, we had to decide how many steps to run gradient descent or how long to run gradient descent, so epochs is a technical term for how many steps of a learning algorithm like gradient descent you may want to run.

Step 1 is to specify the model, which tells TensorFlow how to compute for the inference.
Step 2 compiles the model using a specific loss function.
Step 3 is to train the model.

That's how you can train a neural network in TensorFlow. You are not just call these lines of code to train the model, but that you also understand what's actually going on behind these lines of code, so you don't just call it without really understanding what's going on. This is important because when you're running a learning algorithm, if it doesn't work initially, having that conceptual mental framework of what's really going on will help you debug whenever things don't work the way you expect. 

**Full code:**
```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
  model = Sequential ({
    Dense(units=25, activation='sigmoid')
    Dense(units=15, activation='sigmoid')
    Dense(units=1, activation='sigmoid'))]

from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss=BinaryCrossentropy())

model.fit(X, Y, epochs=100)
```

