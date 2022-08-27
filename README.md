# **Advanced Learning Algorithms**

## **TensorFlow implementation**

This week, we're going to go over training of a neural network. I think being able to take your own data and train your own neural network on it is really fun. We'll look at how you could do that.

### **Train a Neural Network in TensorFlow**

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

## **Training Details**

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

## **Alternatives to the sigmoid activation**

So far, we've been using the sigmoid activation function in all the nodes in the hidden layers and in the output layer. And we have started that way because we were building up neural networks by taking logistic regression and creating a lot of logistic regression units and string them together. But if you use other activation functions, your neural network can become much more powerful.

Recall the demand prediction example where given price, shipping cost, marketing, material, you would try to predict if something is highly affordable. If there's good awareness and high perceived quality and based on that try to predict it was a top seller. But this assumes that awareness is maybe binary is either people are aware or they are not. But it seems like the degree to which possible buyers are aware of the t shirt you're selling may not be binary, they can be a little bit aware, somewhat aware, extremely aware or it could have gone completely viral. So rather than modeling awareness as a binary number `0`, `1`, that you try to estimate the probability of awareness or rather than modeling awareness is just a number between `0` and `1`. Maybe awareness should be any non negative number because there can be any non negative value of awareness going from `0` up to very very large numbers.

So whereas previously we had used a<sub>2</sub><sup>[l]</sup> = g(w<sub>2</sub><sup>[l]</sup>.x + b<sub>2</sub><sup>[l]</sup>) to calculate the activation of that second hidden unit estimating awareness where g was the sigmoid function and just goes between `0` and `1`. If you want to allow a<sub>2</sub><sup>[l]</sup> to potentially take on much larger positive values, we can instead swap in a different activation function. It turns out that a very common choice of activation function in neural networks is this function. It looks like this. It goes if z is this, then g(z) is `0` to the left and then there's this straight line `45°` to the right of `0`. And so when z is greater than or equal to 0, g(z) is just equal to z. That is to the right half of this diagram. And the mathematical equation for this is `g(z) = max(0, z)`. Feel free to verify for yourself that max(0, z) results in this curve. And if a<sub>2</sub><sup>[l]</sup> is g(z) for this value of z, then a, the deactivation value cannot take on 0 or any non negative value.

This activation function has a name. It goes by the name **`ReLU`** with this funny capitalization and **`ReLU`** stands for again, somewhat arcane term, but it stands for _rectified linear unit_. Don't worry too much about what rectified means and what linear unit means. This was just the name that the authors had given to this particular activation function when they came up with it. But most people in deep learning just say **`ReLU`** to refer to this g(z). More generally you have a choice of what to use for g(z) and sometimes we'll use a different choice than the sigmoid activation function. Here are the most commonly used activation functions. You saw the sigmoid activation function, g(z) equals this sigmoid function. On the last slide we just looked at the ReLU or rectified linear unit `g(z) = max(0, z)`. There's one other activation function which is worth mentioning, which is called the linear activation function, which is just g(z) equals to z. Sometimes if you use the linear activation function, people will say we're not using any activation function because if `a=g(z)` where g(z) equals z, then a is just equal to `w.x+b`. And so it's as if there was no g in there at all. So when you are using this linear activation function g(z) sometimes people say, we're not using any activation function. Although in this class, I will refer to using the linear activation function rather than no activation function. But if you hear someone else use that terminology, that's what they mean. It just refers to the linear activation function.

These three (linear, sigmoid, ReLU) are probably by far the most commonly used activation functions in neural networks. We'll touch on the fourth one called the softmax activation function. But with these activation functions you'll be able to build a rich variety of powerful neural networks.

So when building a neural network for each neuron, do you want to use the sigmoid activation function or the ReLU activation function? Or a linear activation function? How do you choose between these different activation functions?


## **Choosing activation functions**

Let's take a look at how you can choose the activation function for different neurons in your neural network. We'll start with some guidance for how to choose it for the output layer. It turns out that depending on what the target label or the ground truth label y is, there will be one fairly natural choice for the activation function for the output layer, and we'll then go and look at the choice of the activation function also for the hidden layers of your neural network.

You can choose different activation functions for different neurons in your neural network, and when considering the activation function for the output layer, it turns out that there'll often be one fairly natural choice, depending on what is the target or the ground truth label y. Specifically, if you are working on a classification problem where y is either `0` or `1`, so a binary classification problem, then the sigmoid activation function will almost always be the most natural choice, because then the neural network learns to predict the probability that `y=1`, just like we had for logistic regression. My recommendation is, if you're working on a binary classification problem, use sigmoid at the output layer.

Alternatively, if you're solving a regression problem, then you might choose a different activation function. For example, if you are trying to predict how tomorrow's stock price will change compared to today's stock price. It can go up or down, and in this case y would be a number that can be either positive or negative, and in that case I would recommend you use the linear activation function. Why is that? Well, that's because then the outputs of your neural network, a<sup>[3]</sup>=f(x) in the example, would be g applied to z<sup>[3]</sup> and with the linear activation function, `g(z)` can take on either positive or negative values. So y can be positive or negative, use a linear activation function.

Finally, if y can only take on non-negative values, such as if you're predicting the price of a house, that can never be negative, then the most natural choice will be the ReLU activation function because as you see here, this activation function only takes on non-negative values, either `0` or positive values. In choosing the activation function to use for your output layer, usually depending on what is the label y you're trying to predict, there'll be one fairly natural choice. In fact, the guidance on this slide is how I pretty much always choose my activation function as well for the output layer of a neural network.

How about the hidden layers of a neural network? It turns out that the ReLU activation function is by far the most common choice in how neural networks are trained by many practitioners today. Even though we had initially described neural networks using the sigmoid activation function, and in fact, in the early history of the development of neural networks, people use sigmoid activation functions in many places, the field has evolved to use ReLU much more often and sigmoids hardly ever. Well, the one exception that you do use a sigmoid activation function in the output layer if you have a binary classification problem.

Why is that? There are a few reasons.

First, if you compare the ReLU and the sigmoid activation functions, the ReLU is a bit faster to compute because it just requires computing `max(0, z)` whereas the sigmoid requires taking an exponentiation and then a inverse and so on, and so it's a little bit less efficient.

The second reason which turns out to be even more important is that the ReLU function goes flat only in one part of the graph; on the left is completely flat, whereas the sigmoid activation function, it goes flat in two places. It goes flat to the left of the graph and it goes flat to the right of the graph. If you're using gradient descent to train a neural network, then when you have a function that is flat in a lot of places, gradient descents would be really slow.

I know that gradient descent optimizes the cost function J(W, B) rather than optimizes the activation function, but the activation function is a piece of what goes into computing, and that results in more places in the cost function J(W, B) that are flats as well and with a small gradient and it slows down learning. I know that that was just an intuitive explanation, but researchers have found that using the ReLU activation function can cause your neural network to learn a bit faster as well, which is why for most practitioners if you're trying to decide what activation functions to use with hidden layer, the ReLU activation function has become now by far the most common choice. In fact that I'm building a neural network, this is how I choose activation functions for the hidden layers as well.

To summarize, here's what I recommend in terms of how you choose the activation functions for your neural network for the output layer:

- Use a sigmoid, if you have a binary classification problem;
- Use linear, if y is a number that can take on positive or negative values,
- Use ReLU if y can take on only positive values or zero positive values or non-negative values.

Then for the hidden layers I would recommend just using ReLU as a default activation function. In TensorFlow, this is how you would implement it:

```
from tf.keras.layers import Dense
model = Sequential([
  Dense(units=25, activation='relu'), ---> first hidden layer
  Dense(units=15, activation='relu'), ---> second hidden layer
  Dense(units=1, activation='sigmoid')
 ])
 ```
 
Rather than saying activation equals sigmoid as we had previously, as TensorFlow to use the ReLU activation function, and then for the output layer in this example, I've asked it to use the sigmoid activation function, but if you wanted to use the linear activation function, is that, that's the syntax for it, or if you wanted to use the ReLU activation function that shows the syntax for it.

With this richer set of activation functions, you'll be well-positioned to build much more powerful neural networks than just once using only the sigmoid activation function. By the way, if you look at the research literature, you sometimes hear of authors using even other activation functions, such as the `tanh` activation function or the `LeakyReLU` activation function or the `swish` activation function. Every few years, researchers sometimes come up with another interesting activation function, and sometimes they do work a little bit better. For example, I've used the `LeakyReLU` activation function a few times in my work, and sometimes it works a little bit better than the ReLU activation function. But I think for the most part, and for the vast majority of applications what you learned about would be good enough. Of course, if you want to learn more about other activation functions, feel free to look on the Internet, and there are just a small handful of cases where these other activation functions could be even more powerful as well.

Why do we even need activation functions at all? Why don't we just use the linear activation function or use no activation function anywhere?

## **Why do we need activation functions?**

Let's take a look at why neural networks need activation functions and why they just don't work if we were to use the linear activation function in every neuron in the neural network.

Recall this demand prediction example. What would happen if we were to use a linear activation function for all of the nodes in this neural network? It turns out that this big neural network will become no different than just linear regression. This would defeat the entire purpose of using a neural network because it would then just not be able to fit anything more complex than the linear regression model.

Let's illustrate this with a simpler example. Let's look at the example of a neural network where the input x is just a number and we have one hidden unit with parameters w<sub>1</sub><sup>[1]</sup> and b<sub>1</sub><sup>[1]</sup> that outputs a<sup>[1]</sup>, just a number, and then the second layer is the output layer and it has also just one output unit with parameters w<sub>1</sub><sup>[2]</sup> and w<sub>1</sub><sup>[2]</sup> and then output a<sup>[1]</sup>, which is also just a number, just a scalar, which is the output of the neural network f(x).

Let's see what this neural network would do if we were to use the linear activation function g(z)=z everywhere. To compute a<sup>[1]</sup> as a function of x, the neural network will use a<sup>[1]</sup> = g(w<sup>[1]</sup>.x + b<sub>1</sub><sup>[1]</sup>). But g(z)=z. So this is just a<sup>[1]</sup> = w<sub>1</sub><sup>[1]</sup>.x + b<sub>1</sub><sup>[1]</sup>.

Then a<sup>[2]</sup> = w<sub>1</sub><sup>[2]</sup>.a<sup>[1]</sup> + b<sub>1</sub><sup>[2]</sup>, because g(z)=z. Let me take this expression for a<sup>[1]</sup> and substitute it in there.

So that becomes a<sup>[2]</sup> = w<sub>1</sub><sup>[2]</sup>(w<sub>1</sub><sup>[1]</sup>.x + b<sub>1</sub><sup>[1]</sup>) + b<sub>1</sub><sup>[2]</sup>.

If we simplify, this becomes a<sup>[2]</sup> = ((w<sub>1</sub><sup>[2]</sup>.w<sub>1</sub><sup>[1]</sup>)x + w<sub>1</sub><sup>[2]</sup>.b<sub>1</sub><sup>[1]</sup>) + b<sub>1</sub><sup>[2]</sup>


It turns out that if I were to set w = w<sub>1</sub><sup>[2]</sup>.w<sub>1</sub><sup>[1]</sup> and set b = w<sub>1</sub><sup>[2]</sup>.b<sub>1</sub><sup>[1]</sup> + b<sub>1</sub><sup>[2]</sup> then what we've just shown is that a<sup>[2]</sup>=w.x+b. So w is just a linear function of the input x. Rather than using a neural network with one hidden layer and one output layer, we might as well have just used a linear regression model. If you're familiar with linear algebra, this result comes from the fact that a linear function of a linear function is itself a linear function. This is why having multiple layers in a neural network doesn't let the neural network compute any more complex features or learn anything more complex than just a linear function.

In the general case, if you had a neural network with multiple layers like this and say you were to use a linear activation function for all of the hidden layers and also use a linear activation function for the output layer, then it turns out this model will compute an output that is completely equivalent to linear regression. The output a<sup>[4]</sup> can be expressed as a linear function of the input features x plus b. Or alternatively, if we were to still use a linear activation function for all the hidden layers, for these three hidden layers here, but we were to use a logistic activation function for the output layer, then it turns out you can show that this model becomes equivalent to logistic regression, and a<sup>[4]</sup>, in this case, can be expressed as 1 over 1 plus e to the negative wx plus b for some values of w and b. So this big neural network doesn't do anything that you can't also do with logistic regression. That's why a common rule of thumb is **don't use the linear activation function in the hidden layers of the neural network**. In fact, I recommend typically using the ReLU activation function should do just fine.

That's why a neural network needs activation functions other than just the linear activation function everywhere. So far, you've learned to build neural networks for binary classification problems where y is either `0` or `1`. As well as for regression problems where y can take negative or positive values, or maybe just positive and non-negative values. 

## **Multiclass**

Multiclass classification refers to classification problems where you can have more than just 2 possible output labels, not just `0` or `1`. Let's take a look at what that means.

For the handwritten digit classification problems, we were just trying to distinguish between the handwritten digits `0` and `1`. But if you're trying to read protocols or zip codes in an envelope, there are actually 10 possible digits you might want to recognize. Or alternatively if you're trying to classify whether patients may have any of 4 or 5 different possible diseases. That too would be a multiclass classification problem or one thing I've worked on a lot is visual defect inspection of parts manufacturer in the factory. Where you might look at the picture of a pill that a pharmaceutical company has manufactured and try to figure out does it have a scratch effect or discoloration defects or a chip defect. This would again be multiple classes of multiple different types of defects that you could classify this pill is having. A multiclass classification problem is still a classification problem in that y you can take on only a small number of discrete categories is not any number, but y can take on more than just 2 possible values.

Whereas previously for buying the classification, you may have had a data set like this with features x<sub>1</sub> and x<sub>2</sub>. In which case logistic regression would fit model to estimate what the probability of y being 1, given the features x. Because y was either 01 with multiclass classification problems, you would instead have a data set that maybe looks like this. Where we have 4 classes where the O's represents one class, the X's represent another class, the triangles represent the third class, and the squares represent the fourth class. Instead of just estimating the chance of y being equal to 1, now want to estimate what's the chance that y is equal to 1, or what's the chance that y is equal to 2? Or what's the chance that y is equal to 3, or the chance of y being equal to 4? It turns out that the algorithm can learn a decision boundary that maybe looks like this that divides the space exploded next to into 4 categories rather than just 2 categories.

That's the definition of the multiclass classification problem. 

## **Softmax**

The softmax regression algorithm is a generalization of logistic regression, which is a binary classification algorithm to the multiclass classification contexts. Let's take a look at how works.

Recall that logistic regression applies when y can take on 2 possible output values, either `0` or `1`, and the way it computes this output is, you would first calculate `z = np.dot(w,x) + b`. Then you would compute a=g(z). We interpreted this as logistic regressions estimates of the probability of y being equal to `1` given those input features x.

If P(y=1|x) = 0.71, what is the P(y=0|x)? The chance of y being the 1, and the chances of y being the 0, they've got to add up to 1, right? There's a 71% chance of it being 1, there has to be a 29% or 0.29 chance of it being equal to 0. To embellish logistic regression a little bit in order to set us up for the generalization to softmax regression, I'm going to think of logistic regression as actually computing two numbers: First a<sub>1</sub> which is this quantity that we had previously of the chance of y being equal to 1 given x, and second, I'm going to think of logistic regression as also computing a<sub>2</sub>, which is 1 minus this which is just the chance of y being equal to zero given the input features x, and so a<sub>1</sub> and a<sub>2</sub> have to add up to 1.

Let's now generalize this to softmax regression, and I'm going to do this with a concrete example of when y can take on 4 possible outputs, so y can take on the values 1, 2, 3 or 4. Here's what softmax regression will do, it will compute `z_1 = np.dot(w_1, x) + b_1`, and then `z_2 = np.dot(w_2, x) + b_2)`, and so on for `z_3 = np.dot(w_2, x) + b_2)` and `z_4 = np.dot(w_2, x) + b_2)`. Here, `w_1`, `w_2`, `w_3`, `w_4` as well as `b_1`, `b_2`, `b_3`, `b_4`, these are the parameters of softmax regression.

Here's the formula for softmax regression, 

# $$a_{1} = \frac{e^{z_{1}}}{e^{z_{1}} + e^{z_{2}} + e^{z_{3}} + e^{z_{4}}} = P(y=1|x)$$

# $$a_{2} = \frac{e^{z_{2}}}{e^{z_{1}} + e^{z_{2}} + e^{z_{3}} + e^{z_{4}}} = P(y=2|x)$$

# $$a_{3} = \frac{e^{z_{3}}}{e^{z_{1}} + e^{z_{2}} + e^{z_{3}} + e^{z_{4}}} = P(y=3|x)$$

# $$a_{4} = \frac{e^{z_{4}}}{e^{z_{1}} + e^{z_{2}} + e^{z_{3}} + e^{z_{4}}} = P(y=4|x)$$


Whereas on the left, we wrote down the specification for the logistic regression model, these equations on the right are our specification for the softmax regression model. It has parameters w_1 through w_4, and b_1 through b_4, and if you can learn appropriate choices to all these parameters, then this gives you a way of predicting what's the chance of y being 1, 2, 3 or 4, given a set of input features x.

Let's see, run softmax regression on a new input x, and you find that a<sub>1</sub> is 0.30, a<sub>2</sub> is 0.20, a<sub>3</sub> is 0.15. What do you think a<sub>4</sub> will be? Because the chance of y take on the values of 1, 2, 3 or 4, they have to add up to one, a<sub>4</sub> the chance of y being with a four has to be `0.35`, which is `1-(0.30)-(0.20)-(0.15)`. Here I wrote down the formulas for softmax regression in the case of four possible outputs, and let's now write down the formula for the general case for softmax regression.

In the general case, y can take on n possible values, so y can be 1, 2, 3, and so on up to n. In that case, softmax regression will compute to `z_j = np.dot(w_j, x) + b_j)`, where the parameters of softmax regression are w<sub>1</sub>, w<sub>2</sub> through to w<sub>N</sub>, as well as b<sub>1</sub>, b<sub>2</sub> through b<sub>N</sub>. Finally, we'll compute



# $$\frac{e^{z_{j}} {\!\sum_{k=1}^{N}e^{z_{k}}$$


a j equals e to the z j divided by sum from k equals 1 to n of e to the z sub k. While here I'm using another variable k to index the summation because here j refers to a specific fixed number like j equals 1. A, j is interpreted as the model's estimate that y is equal to j given the input features x. Notice that by construction that this formula, if you add up a1, a2 all the way through a n, these numbers always will end up adding up to 1. We specified how you would compute the softmax regression model. I won't prove it in this video, but it turns out that if you apply softmax regression with n equals 2, so there are only two possible output classes then softmax regression ends up computing basically the same thing as logistic regression. The parameters end up being a little bit different, but it ends up reducing to logistic regression model. But that's why the softmax regression model is the generalization of logistic regression. Having defined how softmax regression computes it's outputs, let's now take a look at how to specify the cost function for softmax regression. Recall for logistic regression, this is what we had. We said z is equal to this. Then I wrote earlier that a1 is g of z, was interpreted as a probability of y is 1. We also wrote a2 is the probability that y is equal to 0. Previously, we had written the loss of logistic regression as negative y log a1 minus 1 minus y log 1 minus a1. But 1 minus a1 is also equal to a2, because a2 is one minus a1 according to this expression over here. I can rewrite or simplify the loss for logistic regression little bit to be negative y log a1 minus 1 minus y log of a2. In other words, the loss if y is equal to 1 is negative log a1. If y is equal to 0, then the loss is negative log a2, and then same as before the cost function for all the parameters in the model is the average loss, average over the entire training set. That was a cost function for this regression. Let's write down the cost function that is conventionally use the softmax regression. Recall that these are the equations we use for softmax regression. The loss we're going to use for softmax regression is just this. The loss for if the algorithm puts a1 through an. The ground truth label is why is if y equals 1, the loss is negative log a1. Says negative log of the probability that it thought y was equal to 1, or if y is equal to 2, then I'm going to define as negative log a2. Y is equal to 2. The loss of the algorithm on this example is negative log of the probability it's thought y was equal to 2. On all the way down to if y is equal to n, then the loss is negative log of a n. To illustrate what this is doing, if y is equal to j, then the loss is negative log of a j. That's what this function looks like. Negative log of a j is a curve that looks like this. If a j was very close to 1, then you beyond this part of the curve and the loss will be very small. But if it thought, say, a j had only a 50% chance then the loss gets a little bit bigger. The smaller a j is, the bigger the loss. This incentivizes the algorithm to make a j as large as possible, as close to 1 as possible. Because whatever the actual value y was, you want the algorithm to say hopefully that the chance of y being that value was pretty large. Notice that in this loss function, y in each training example can take on only one value. You end up computing this negative log of a j only for one value of a j, which is whatever was the actual value of y equals j in that particular training example. For example, if y was equal to 2, you end up computing negative log of a2, but not any of the other negative log of a1 or the other terms here. That's the form of the model as well as the cost function for softmax regression. If you were to train this model, you can start to build multiclass classification algorithms. What we'd like to do next is take this softmax regression model, and fit it into a new network so that you really do something even better, which is to train a new network for multi-class classification. Let's go through that in the next video.





































