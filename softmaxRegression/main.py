import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession()  # 将session注册为默认的session
x = tf.placeholder(tf.float32, [None, 784])  # 创建一个placeholder，即输入数据的地方，第一个参数是数据类型，第二个参数是tesor的shape（数据的尺寸）
W = tf.Variable(tf.zeros([784, 10]))  # 创建variable变量，因为这个变量是可以持续化的。
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax 回归
# 自动实现了正向和方向传播，只需定义好loss，训练时会自动求导并进行梯度下降，完成softmax回归模型参数的自动学习

# 定义loss
y_ = tf.placeholder(tf.float32, [None, 10])  # 输入数据是真实的label
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # reduce_mean对每个batch求均值，reduce_sum累计求和

# 定义优化算法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 全局初始化
init = tf.global_variables_initializer()

init.run()

# 迭代执行训练操作，每次随机从训练集中抽取100条样本构成一个mini-batch，并feed给placeholder
for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# 准确率验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # tf.argmax是从一个tensor中寻找最大值的序号，1表示轴  向量中的值是bool型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 将bool转float32，在求均值(0,1)求均值就是精确度


print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
