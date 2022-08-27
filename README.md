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

### **Training Details**

Before looking at the details of training in neural network, recall how you had trained a logistic regression model.

Step one of building a logistic regression model was you would **specify how to compute the output given the input feature x and the parameters w and b**. The logistic regression function predicts $f(x) = G$. The sigmoid function applied to $W.X + B$ which was the sigmoid function applied to $W.X + B$. If `z = np.dot(w,x) + b`, then `f_x = 1/(1+np.exp(-z))`, so those first step were to specify what is the input to output function of logistic regression, and that depends on both the input x and the parameters of the model.

The second step we had to do to train the literacy regression model was to **specify the loss function and also the cost function**, so the loss function said, if logistic regression outputs f(x) and the ground truth label, the actual label and a training set was y then the loss on that single training example was `loss = -y * np.log(f_x) - (1-y) * np.log(1-f_x)`. This was a measure of how well is logistic regression doing on a single training example `(x,y)`. Given this definition of a loss function, we then define the cost function, and the cost function was a function of the parameters w and b, and that was just the average that is taking an average overall M training examples of the loss function computed on the M training examples, X<sub>1</sub>, Y<sub>1</sub> through X<sub>M</sub>Y<sub>M</sub>, and remember that in the convention we're using the loss function is a function of the output of the learning algorithm and the ground truth label as computed over a single training example whereas the cost function J is an average of the loss function computed over your entire training set. That was step two of what we did when building up logistic regression.

The third and final step to train a logistic regression model was to use an algorithm specifically gradient descent to minimize that cost function J of w,b to minimize it as a function of the parameters w and b. We minimize the cost J as a function of the parameters using gradient descent where w is updated as `w = w - alpha * dj_dw`. b similarly is updated as `b = b - alpha * dj_db`. To summarize:

_Step 1_: Specifying how to compute the outputs given the input x and parameters.
_Step 2_: Specify loss and costs.
_Step 3_: Minimize the cost function we trained logistic regression.

The same 3 steps is how we can train a **neural network in TensorFlow**.

Step 1 is **specify how to compute the output given the input x and parameters w and b that's done with this code snippet and this was actually enough to specify the computations needed in forward propagation or for the inference algorithm for example.

```
model = Sequential([
  Dense(...)
  Dense(...)
  Dense(...)
    ])
```

The second step is **to compile the model and to tell it what loss you want to use**, and here's the code that you use to specify this loss function which is the binary cross entropy loss function, and once you specify this loss taking an average over the entire training set also gives you the cost function for the neural network.

```
model.compile(loss = BinaryCrossentropy())
```

The step 3 is to **call function to try to minimize the cost as a function of the parameters of the neural network**.

```
model.fit(X, y, epochs=100)
```

Let's look in greater detail in these three steps in the context of training a neural network.

The first step, **specify how to compute the output given the input x and parameters w and b**. This code snippet specifies the entire architecture of the neural network. It tells you that there are 25 hidden units in the first hidden layer, then the 15 in the next one, and then one output unit and that we're using the sigmoid activation value. Based on this code snippet, we know also what are the parameters w<sub>1</sub>, b<sub>1</sub> though the first layer parameters of the second layer and parameters of the third layer. 

```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
  Dense(units=25, activation='sigmoid')
  Dense(units=15, activation='sigmoid')
  Dense(units=1, activation='sigmoid')
```


This code specifies the entire architecture of the neural network and therefore tells TensorFlow everything it needs in order to compute the output x as a function. In order to compute the output a<sup>3</sup> or f of x as a function of the input x and the parameters, here we have written w<sub>l</sub> and b<sub>l</sub>.

In the second step, you have to **specify what is the loss function**. That will also define the cost function we use to train the neural network. For the MNIST 01 digit classification problem is a binary classification problem and most common by far, loss function to use is this one is actually the same loss function as what we had for logistic regression is `L(f(x), y) = -ylog(f(x)) - (1-y)log(1-f(x))` where y is the ground truth label, sometimes also called the target label y, and `f(x)` is now the output of the neural network.

In the terminology of TensorFlow, this loss function is called **binary cross entropy**. The syntax is to ask TensorFlow to compile the neural network using this loss function. Another historical note, Keras was originally a library that had developed independently of TensorFlow is actually totally separate project from TensorFlow. But eventually it got merged into TensorFlow, which is why we have tf.Keras library.losses dot the name of this loss function. Having specified the loss with respect to a single training example, TensorFlow knows that it costs you want to minimize is then the average, taking the average over all m training examples of the loss on all of the training examples. Optimizing this cost function will result in fitting the neural network to your binary classification data.

In case you want to solve a regression problem rather than a classification problem. You can also tell TensorFlow to compile your model using a different loss function. For example, if you have a regression problem and if you want to minimize the squared error loss. Here is the squared error loss: `from tensorflow.keras.losses import MeanSquaredError` The loss with respect to if your learning algorithm outputs f(x) with a target or ground truth label of y, that's 1/2 of the squared error. Then you can use this loss function in TensorFlow, which is to use the maybe more intuitively named mean squared error loss function. Then TensorFlow will try to minimize the mean squared error.

In this expression, I'm using J(W, B) b to denote the cost function. The cost function is a function of all the parameters into neural network. You can think of W as including W<sub>1</sub>, W<sub>2</sub>, W<sub>3</sub>. All the W parameters and the entire new network and be as including B<sub>1</sub>, B<sub>2</sub>, and B<sub>3</sub>. If you are optimizing the cost function respect to W and B, if we tried to optimize it with respect to all of the parameters in the neural network. Up on top as well, I had written f(x) as the output of the neural network, but we can also write f<sub>W, B</sub> if we want to emphasize that the output of the neural network as a function of x depends on all the parameters in all the layers of the neural network. That's the loss function and the cost function. In TensorFlow, this is called **the binary cross-entropy loss function**.

Where does that name come from? It turns out in statistics this function on top is called the cross-entropy loss function, so that's what cross-entropy means, and the word binary just reemphasizes or points out that this is a binary classification problem because each image is either 0 or 1.

Finally, you will ask TensorFlow to minimize the cross-function. You might remember the gradient descent algorithm from the first course. If you're using gradient descent to train the parameters of a neural network, then you are repeatedly, for every layer l and for every unit j, update w<sub>j</sub><sup>l</sup> according to w<sub>j</sub><sup>l</sup> - the learning rate alpha * the partial derivative with respect to that parameter of the cost function J<sub>w, b</sub> and similarly for the parameters b as well.

After doing, say, 100 iterations of gradient descent, hopefully, you get to a good value of the parameters. In order to use gradient descent, the key thing you need to compute is these partial derivative terms. What TensorFlow does, and, in fact, what is standard in neural network training, is to use an algorithm called _backpropagation_ in order to compute these partial derivative terms. TensorFlow can do all of these things for you. It implements backpropagation all within this function called fit. All you have to do is `model.fit(X, y, epochs=100)`. In fact, TensorFlow can use an algorithm that is even a little bit faster than gradient descent.

Now, I know that we're relying heavily on the TensorFlow library in order to implement a neural network. One pattern I've seen across multiple ideas is as the technology evolves, libraries become more mature, and most engineers will use libraries rather than implement code from scratch. There have been many other examples of this in the history of computing. Once, many decades ago, programmers had to implement their own sorting function from scratch, but now sorting libraries are quite mature that you probably call someone else's sorting function rather than implement it yourself, unless you're taking a computing class and I ask you to do it as an exercise.

Today, if you want to compute the square root of a number, like what is the square root of seven, well, once programmers had to write their own code to compute this, but now pretty much everyone just calls a library to take square roots, or matrix operations, such as multiplying two matrices together. When deep learning was younger and less mature, many developers were implementing things from scratch using Python or C++ or some other library. But today, deep learning libraries have matured enough that most developers will use these libraries, and, in fact, most commercial implementations of neural networks today use a library like TensorFlow or PyTorch. But as I've mentioned, it's still useful to understand how they work under the hood so that if something unexpected happens, which still does with today's libraries, you have a better chance of knowing how to fix it.

### **Alternatives to the sigmoid activation**

So far, we've been using the sigmoid activation function in all the nodes in the hidden layers and in the output layer. And we have started that way because we were building up neural networks by taking logistic regression and creating a lot of logistic regression units and string them together. But if you use other activation functions, your neural network can become much more powerful.

Recall the demand prediction example where given price, shipping cost, marketing, material, you would try to predict if something is highly affordable. If there's good awareness and high perceived quality and based on that try to predict it was a top seller. But this assumes that awareness is maybe binary is either people are aware or they are not. But it seems like the degree to which possible buyers are aware of the t shirt you're selling may not be binary, they can be a little bit aware, somewhat aware, extremely aware or it could have gone completely viral. So rather than modeling awareness as a binary number `0`, `1`, that you try to estimate the probability of awareness or rather than modeling awareness is just a number between `0` and `1`. Maybe awareness should be any non negative number because there can be any non negative value of awareness going from `0` up to very very large numbers.

So whereas previously we had used a<sub>2</sub><sup>[l]</sup> = g(w<sub>2</sub><sup>[l]</sup>.x + b<sub>2</sub><sup>[l]</sup>) to calculate the activation of that second hidden unit estimating awareness where g was the sigmoid function and just goes between `0` and `1`. If you want to allow a<sub>2</sub><sup>[l]</sup> to potentially take on much larger positive values, we can instead swap in a different activation function. It turns out that a very common choice of activation function in neural networks is this function. It looks like this. It goes if z is this, then g(z) is `0` to the left and then there's this straight line `45°` to the right of `0`. And so when z is greater than or equal to 0, g(z) is just equal to z. That is to the right half of this diagram. And the mathematical equation for this is `g(z) = max(0, z)`. Feel free to verify for yourself that max(0, z) results in this curve. And if a<sub>2</sub><sup>[l]</sup> is g(z) for this value of z, then a, the deactivation value cannot take on 0 or any non negative value.

This activation function has a name. It goes by the name **`ReLU`** with this funny capitalization and **`ReLU`** stands for again, somewhat arcane term, but it stands for _rectified linear unit_. Don't worry too much about what rectified means and what linear unit means. This was just the name that the authors had given to this particular activation function when they came up with it. But most people in deep learning just say **`ReLU`** to refer to this g(z). More generally you have a choice of what to use for g(z) and sometimes we'll use a different choice than the sigmoid activation function. Here are the most commonly used activation functions. You saw the sigmoid activation function, g(z) equals this sigmoid function. On the last slide we just looked at the ReLU or rectified linear unit `g(z) = max(0, z)`. There's one other activation function which is worth mentioning, which is called the linear activation function, which is just g(z) equals to z. Sometimes if you use the linear activation function, people will say we're not using any activation function because if `a=g(z)` where g(z) equals z, then a is just equal to `w.x+b`. And so it's as if there was no g in there at all. So when you are using this linear activation function g(z) sometimes people say, we're not using any activation function. Although in this class, I will refer to using the linear activation function rather than no activation function. But if you hear someone else use that terminology, that's what they mean. It just refers to the linear activation function.

These three (linear, sigmoid, ReLU) are probably by far the most commonly used activation functions in neural networks. We'll touch on the fourth one called the softmax activation function. But with these activation functions you'll be able to build a rich variety of powerful neural networks.

So when building a neural network for each neuron, do you want to use the sigmoid activation function or the ReLU activation function? Or a linear activation function? How do you choose between these different activation functions?












