"""
@Author:Che_Hongshu
@Function: tools for CNN-CIFAR-10 dataset
@Modify:2018.3.5
@IDE: pycharm
@python :3.6
@os : win10
"""

import tensorflow as tf
"""
函数说明: 得到weights变量和weights的loss
Parameters:
   shape-维度
   stddev-方差
   w1-
Returns:
    var-维度为shape，方差为stddev变量
CSDN:
    http://blog.csdn.net/qq_33431368
Modify:
    2018-3-5
"""
def variable_with_weight_loss(shape, stddev, w1):
    """
    tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    从截断的正态分布中输出随机值。 shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产
    生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与
    均值的差值大于两倍的标准差，那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随
    机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
    :param shape:
    :param stddev:
    :param w1:
    :return:
    """
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var
"""
函数说明: 得到总体的losses
Parameters:
   logits-通过神经网络之后的前向传播的结果
   labels-图片的标签
Returns:
   losses
CSDN:
    http://blog.csdn.net/qq_33431368
Modify:
    2018-3-5
"""
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
        (logits=logits, labels=labels, name='total_loss')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entorpy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

"""
函数说明: 对变量进行min max 和 stddev的tensorboard显示
Parameters:
    var-变量
    name-名字
Returns:
    None
CSDN:
    http://blog.csdn.net/qq_33431368
Modify:
    2018-3-5
"""
def variables_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var-mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)
        # tf.summary.histogram()
