from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
ssess = tf.InteractiveSession()

in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # reduce_mean对每个batch求均值，reduce_sum累计求和
train_step = tf.train.AdagradOptimizer(0.30).minimize(cross_entropy)

init = tf.global_variables_initializer()
init.run()
for i in range(4000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# 准确率验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # tf.argmax是从一个tensor中寻找最大值的序号，1表示轴  向量中的值是bool型
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 将bool转float32，在求均值(0,1)求均值就是精确度

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
