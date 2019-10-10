---
layout:     post
title:      人脸识别的常用loss及tensorflow实现
subtitle:   常用loss总结
date:       2019-08-25
author:     Jieson
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - tensorflow
    - face recognition
---
### <center>人脸识别的常用loss及tensorflow实现</center>
&#160;&#160;&#160;&#160;在人脸识别中，模型的提升主要体现在损失函数的设计上，损失函数会对整个网络的优化有着导向性的作用。从传统的softmax loss到cosface, arcface 都有这一定的提高。

1、softmax loss

$$
loss = -\frac{1}{m}\sum_{i=0}^m log\frac{e^{W^T_{y_i} + b_{y_i}}}{\sum_{j=1}^N e^{W^T_{j}+b_j}}
$$

![softmax](https://note.youdao.com/yws/api/personal/file/WEB609552c3c1972fbc40642cf3f7c752e2?method=download&shareKey=87d9648040232ac180ce0f5130bba49a)

&#160;&#160;&#160;&#160;softmax 只考虑了是否正确分类，但没有考虑类间距离，Softmax并不要求类内紧凑和类间分离，这一点非常不适合人脸识别任务。所以需要改造Softmax，除了保证可分性外，还要做到特征向量类内尽可能紧凑，类间尽可能分离。
```
tf.nn.softmax_cross_entropy_with_logits
```
2、center loss

$$
L_C = -\frac{1}{2}\sum_{i=1}^m||x_i-C_{y_i}||^2
$$

![center](https://note.youdao.com/yws/api/personal/file/WEBa45702108fbbc3a07eba23767681bc73?method=download&shareKey=4a45d57af0cd2593c315adfbac1be3c6)

&#160;&#160;&#160;&#160;center loss 考虑到不仅仅是分类要对，而且要求类间有一定的距离。上面的公式中
\\(\large C_{y_i}\\)表示某一类的中心,\\(x_i\\) 表示每个人脸的特征值。作者在softmax loss的基础上加入了\\(L_C\\)，同时使用参数\\(lambda\\)来控制类内距离，整体的损失函数如下：

$$
L = L_S + L_C = -\frac{1}{m}\sum_{i=0}^m log\frac{e^{W^T_{y_i}} + b_{y_i}}{\sum_{j=1}^N e^{W^T_{j}+b_j}} + \frac{1}{2}\sum_{i=1}^m||x_i-C_{y_i}||^2
$$

```
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers
```

3、Triplet Loss

![triplet](https://note.youdao.com/yws/api/personal/file/WEB178d4497e7a1410d86ab55d074c8c2a1?method=download&shareKey=2e40a3cb957c7adcc6ef75577f974c4e)

&#160;&#160;&#160;&#160;三元组损失函数，三元组由Anchor、Negative、Positive组成，从上图可以看到，triplet loss 就是使同类距离更近，类间更加远离。

$$
tripletloss = \sum_{i}^N[||f(x_{i}^a) - f(x_{i}^p)||^2 - ||f(x_{i}^a) - f(x_{i}^n)||^2 + \alpha]
$$

表达第一项为类内距离，中间项为类间距离，\\(\alpha\\)为margin。使用梯度下降法优化就是使类内距离不断下降，类间距离不断增大。

优点：直接使用embeddings计算相似度作为loss，加大类间距离，压缩类内间距

缺点：训练收敛慢，对triplet 对的选取比较敏感

```
def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss
```

4、Arcface
前面的softmax Loss 没有考虑类间距离， center loss 学习类中心，使类内紧凑，但没有类间可分。triplet loss 收敛较慢。因此就产生了sofmax的变形loss，如L-Softmax、SphereFace、Arcface。arcface是直接在角度空间中最大化分类界限，而cosface是在余弦空间中最大化分类界限，角度距离比余弦距离在对角度的影响更加直接。

![arcface](https://note.youdao.com/yws/api/personal/file/WEB7137d913f307c97ced07c50f96b4f2ef?method=download&shareKey=d572ca0285212f680612e6df0e8e53d7)

$$
arcface = - \frac{1}{N}\sum_{i=1}^Nlog(\frac{e^{s(cos(\theta_{yi}+m))}}{e^s(cos(\theta_{yi}+m)) + \sum_{j=1, {j} \neq {y_i}}e^{scos(\theta_j)}})
$$

```
def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.45):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output
```

### reference
> [1]https://blog.csdn.net/u012505617/article/details/89355690<br>
> [2]http://ydwen.github.io/papers/WenECCV16.pdf<br>
> [3]Wen Y, Zhang K, Li Z, et al. A discriminative feature learning approach for deep face recognition [C]// ECCV, 2016.<br>
> [3]Liu W, Wen Y, Yu Z, et al. Large-Margin Softmax Loss for Convolutional Neural Networks [C]// ICML, 2016.<br>
> [4]Liu W, Wen Y, Yu Z, et al. SphereFace: Deep Hypersphere Embedding for Face Recognition [C]// CVPR. 2017.<br>
> [5]https://arxiv.org/abs/1801.07698<br>
> [6]https://github.com/deepinsight/insightface<br>
> [7]https://github.com/davidsandberg/facenet<br>
