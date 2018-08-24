## TensorFlow.js中的核心概念

TensorFlow.js 是一个用户机器智能的开源WEBGL加速JavaScript库. 它为您带来了高性能的机器学习构建模块, 允许您在浏览器中训练神经网络或在推断模式下运行预先训练的模型. 有关安装/配置TensorFlow.js的指南，请参阅 Getting Started。

TensorFlow.js为机器学习提供了低级和高级别的构建块, 基于Keras的API，用于构建神经网络.我们来看看该库的一些核心组件.


## 张量

TensorFlow.js中的核心是张量: 一组数值，形状为一个或多个维度的数组. Tensor实例具有定义数组形状的形状属性（即，数组的每个维度中有多少个值）.

Tensor的主要构造函数是tf.tensor函数：

```
`javascript
`
// 2x3 Tensor
const shape = [2, 3]; // 2 rows, 3 columns
const a = tf.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape);
a.print(); // print Tensor values
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]

// The shape can also be inferred:
const b = tf.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
b.print();
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]

```

但是，为了构建低秩张量，我们建议使用以下函数来增强代码可读性: 

tf.scalar, tf.tensor1d, tf.tensor2d, tf.tensor3d and tf.tensor4d.

以下示例使用tf.tensor2d创建与上面相同的张量：

```
const c = tf.tensor2d([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
c.print();
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]

```

TensorFlow.js还提供了方便函数，用于创建所有值设置为0（tf.zeros）或所有值设置为1（tf.ones）的张量：

```
// 3x5 Tensor with all values set to 0
const zeros = tf.zeros([3, 5]);
// Output: [[0, 0, 0, 0, 0],
//          [0, 0, 0, 0, 0],
//          [0, 0, 0, 0, 0]]

```

在TensorFlow.js中，张量是不可变的;一旦创建，您就无法更改其值.而是对它们执行生成新张量的操作.

## 变量

变量用一个张量的值初始化.然而,与张量不同,其值是可变的. 您可以使用assign方法为现有变量指定新的张量:

```
const initialValues = tf.zeros([5]);
const biases = tf.variable(initialValues); // initialize biases
biases.print(); // output: [0, 0, 0, 0, 0]

const updatedValues = tf.tensor1d([0, 1, 0, 1, 0]);
biases.assign(updatedValues); // update values of biases
biases.print(); // output: [0, 1, 0, 1, 0]

```

变量主要用于在模型训练期间存储然后更新值。

## 操作 (Ops)

张量允许您存储数据, operations（ops）允许您操作该数据. TensorFlow.js提供了多种适用于线性代数和机器学习的运算，可以在张量上执行. 因为张量是不可变的, 这些操作不会改变它们的价值观;相反，ops会返回新的张量.

可用的操作包括一元操作，如square:

```

const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const d_squared = d.square();
d_squared.print();
// Output: [[1, 4 ],
//          [9, 16]]

```

和二元操作，如add，sub和mul:

```

const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);

const e_plus_f = e.add(f);
e_plus_f.print();
// Output: [[6 , 8 ],
//          [10, 12]]

```

TensorFlow.js有一个支持链式调用的接口;你可以在ops的结果上调用ops：

```
const sq_sum = e.add(f).square();
sq_sum.print();
// Output: [[36 , 64 ],
//          [100, 144]]

// All operations are also exposed as functions in the main namespace,
// so you could also do the following:
const sq_sum = tf.square(tf.add(e, f));

```

## 模型和层

从概念上讲，模型是一种函数，给定一些输入将产生一些所需的输出.

在TensorFlow.js中，有两种方法可以创建模型. 您可以直接使用ops来表示模型所做的工作. 例如:

```

// Define function
function predict(input) {
  // y = a * x ^ 2 + b * x + c
  // More on tf.tidy in the next section
  return tf.tidy(() => {
    const x = tf.scalar(input);

    const ax2 = a.mul(x.square());
    const bx = b.mul(x);
    const y = ax2.add(bx).add(c);

    return y;
  });
}

// Define constants: y = 2x^2 + 4x + 8
const a = tf.scalar(2);
const b = tf.scalar(4);
const c = tf.scalar(8);

// Predict output for input of 2
const result = predict(2);
result.print() // Output: 24

```

您还可以使用高级API tf.model来构建模型层,这是深度学习中流行的抽象.以下代码构造了一个tf.sequential模型:


```

const model = tf.sequential();
model.add(
  tf.layers.simpleRNN({
    units: 20,
    recurrentInitializer: 'GlorotNormal',
    inputShape: [80, 4]
  })
);

const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({optimizer, loss: 'categoricalCrossentropy'});
model.fit({x: data, y: labels});

```

TensorFlow.js中有许多不同类型的层 一些示例包括tf.layers.simpleRNN，tf.layers.gru和tf.layers.lstm.

## 内存管理：dispose和tf.tidy

由于TensorFlow.js使用GPU来加速数学运算，因此在使用张量和变量时需要管理GPU内存.

TensorFlow.js提供了两个函数来帮助解决这个问题：dispose和tf.tidy.

## dispose

您可以在张量或变量上调用dispose来清除它并释放其GPU内存:

```

const x = tf.tensor2d([[0.0, 2.0], [4.0, 6.0]]);
const x_squared = x.square();

x.dispose();
x_squared.dispose();

```

## tf.tidy

在进行大量张量操作时，使用dispose会很麻烦.TensorFlow.js提供了另一个功能, tf.tidy, 它与JavaScript中的常规范围起着类似的作用,但作用于GPU上的张量.

tf.tidy执行一个函数并清除所创建的任何中间张量，释放它们的GPU内存.它不会清除内部函数的返回值.

```

// tf.tidy takes a function to tidy up after
const average = tf.tidy(() => {
  // tf.tidy will clean up all the GPU memory used by tensors inside
  // this function, other than the tensor that is returned.
  //
  // Even in a short sequence of operations like the one below, a number
  // of intermediate tensors get created. So it is a good practice to
  // put your math ops in a tidy!
  const y = tf.tensor1d([1.0, 2.0, 3.0, 4.0]);
  const z = tf.ones([4]);

  return y.sub(z).square().mean();
});

average.print() // Output: 3.5

```

使用tf.tidy将有助于防止应用程序中的内存泄漏. 它还可以用于更加谨慎地控制何时回收内存.


## 两个重要的注释

传递给tf.tidy的函数应该是同步的，也不会返回Promise. 我们建议保留更新UI的代码或在tf.tidy之外发出远程请求.

tf.tidy不会清理变量. 变量通常持续到机器学习模型的整个生命周期, 所以即使它们是整洁的TensorFlow.js也不会清理它们;但是，您可以手动调用dispose清理它们.

## 其他资源

有关库的综合文档，请参阅TensorFlow.js API参考。

要更深入地了解机器学习基础知识，请参阅以下资源：

机器学习速成课程(注意：本课程的练习使用TensorFlow的Python API. 但是，它所教授的核心机器学习概念可以使用TensorFlow.js以相同的方式应用.)

机器学习术语表
