## 训练第一步：拟合曲线到合成数据

本教程演示了如何使用TensorFlow.js运算符重头构建小型玩具模型. 我们将使用多项式函数生成一些合成数据的曲线.

## 先决条件

本教程假设您熟悉TensorFlow.js的基本构建块介绍在[TensorFlow.js中的核心概念](./core-concepts.md):张量,变量和操作.我们建议在完成本教程之前完成核心概念。

## 运行代码
本教程重点介绍用于构建模型和其学习率的TensorFlow.js代码.可以在[此处](https://github.com/tensorflow/tfjs-examples/tree/master/polynomial-regression-core)找到本教程的完整代码（包括数据生成和图表绘图代码）.

要在本地运行代码，需要安装以下依赖项:

Node.js version 8.9 or higher

Yarn or NPM CLI

这些说明使用Yarn，但是如果您熟悉NPM CLI并且更喜欢使用它，那么它仍然可以使用.

$ git clone https://github.com/tensorflow/tfjs-examples
$ cd tfjs-examples/polynomial-regression-core
$ yarn
$ yarn watch
The tfjs-examples/polynomial-regression-core directory above is completely standalone so you can copy it to start your own project.

Input Data
Our synthetic data set is composed of x- and y-coordinates that look as follows when plotted on a Cartesian plane:

Input data scatterplot. Data approximates a cubic function with a local minimum around (-0.6, 0) and a local maximum around (0.4, 1)
This data was generated using a cubic function of the format y = ax3 + bx2 + cx + d.

Our task is to learn the coefficients of this function: the values of a, b, c, and d that best fit the data. Let's take a look at how we might learn those values using TensorFlow.js operations.

Step 1: Set up Variables
First, let's create some variables to hold our current best estimate of these values at each step of model training. To start, we'll assign each of these variables a random number:

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));
Step 2: Build a Model
We can represent our polynomial function y = ax3 + bx2 + cx + d in TensorFlow.js by chaining a series of mathematical operations: addition (add), multiplication (mul), and exponentiation (pow and square).

The following code constructs a predict function that takes x as input and returns y:

function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3))) // a * x^3
      .add(b.mul(x.square())) // + b * x ^ 2
      .add(c.mul(x)) // + c * x
      .add(d); // + d
  });
}
Let's go ahead and plot our polynomial function using the random values for a, b, c, and d that we set in Step 1. Our plot will likely look something like this:

Cubic function that poorly fits the data in the previous graph. The function hovers far above the data from x=-1.0 to x=0, and then zooms upward from x=0.2 to x=1.0, while the data points move downward.
Because we started with random values, our function is likely a very poor fit for the data set. The model has yet to learn better values for the coefficients.

Step 3: Train the Model
Our final step is to train the model to learn good values for the coefficients. To train our model, we need to define three things:

A loss function, which measures how well a given polynomial fits the data. The lower the loss value, the better the polynomial fits the data.

An optimizer, which implements an algorithm for revising our coefficient values based on the output of the loss function. The optimizer's goal is to minimize the output value of the loss function.

A training loop, which will iteratively run the optimizer to minimize loss.

Define the Loss Function
For this tutorial, we'll use mean squared error (MSE) as our loss function. MSE is calculated by squaring the difference between the actual y value and the predicted y value for each x value in our data set, and then taking the mean of all the resulting terms.

We can define a MSE loss function in TensorFlow.js as follows:

function loss(predictions, labels) {
  // Subtract our labels (actual values) from predictions, square the results,
  // and take the mean.
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}
Define the Optimizer
For our optimizer, we'll use Stochastic Gradient Descent (SGD). SGD works by taking the gradient of a random point in our data set and using its value to inform whether to increase or decrease the value of our model coefficients.

TensorFlow.js provides a convenience function for performing SGD, so that you don't have to worry about performing all these mathematical operations yourself. tf.train.sgd takes as input a desired learning rate, and returns an SGDOptimizer object, which can be invoked to optimize the value of the loss function.

The learning rate controls how big the model's adjustments will be when improving its predictions. A low learning rate will make the learning process run more slowly (more training iterations needed to learn good coefficients), while a high learning rate will speed up learning but might result in the model oscillating around the right values, always overcorrecting.

The following code constructs an SGD optimizer with a learning rate of 0.5:

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);
Define the Training Loop
Now that we've defined our loss function and optimizer, we can build a training loop, which iteratively performs SGD to refine our model's coefficients to minimize loss (MSE). Here's what our loop looks like:

function train(xs, ys, numIterations = 75) {

  const learningRate = 0.5;
  const optimizer = tf.train.sgd(learningRate);

  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      return loss(predsYs, ys);
    });
  }
}
Let's take a closer look at the code, step by step. First, we define our training function to take the x and y values of our dataset, as well as a specified number of iterations, as input:

function train(xs, ys, numIterations) {
...
}
Next, we define the learning rate and SGD optimizer as discussed in the previous section:

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);
Finally, we set up a for loop that runs numIterations training iterations. In each iteration, we invoke minimize on the optimizer, which is where the magic happens:

for (let iter = 0; iter < numIterations; iter++) {
  optimizer.minimize(() => {
    const predsYs = predict(xs);
    return loss(predsYs, ys);
  });
}
minimize takes a function that does two things:

It predicts y values (predYs) for all the x values using the predict model function we defined earlier in Step 2.

It returns the mean squared error loss for those predictions using the loss function we defined earlier in Define the Loss Function.

minimize then automatically adjusts any Variables used by this function (here, the coefficients a, b, c, and d) in order to minimize the return value (our loss).

After running our training loop, a, b, c, and d will contain the coefficient values learned by the model after 75 iterations of SGD.

See the Results!
Once the program finishes running, we can take the final values of our variables a, b, c, and d, and use them to plot a curve:

A cubic curve that nicely approximates the shape of our data
The result is much better than the curve we originally plotted using random values for the coefficient.

Additional Resources
See Core Concepts in TensorFlow.js for an introduction to the core building blocks in TensorFlow.js: tensors, variables, and ops.

See Descending into ML in Machine Learning Crash Course for a more in-depth introduction to machine learning loss

See Reducing Loss in Machine Learning Crash Course for a deeper dive into gradient descent and SGD.
