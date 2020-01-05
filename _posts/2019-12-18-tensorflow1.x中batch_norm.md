---
layout:     post
title:      tensorflow1.x中batch_norm
subtitle:   避坑操作
date:       2019-12-18
author:     Jieson
header-img: img/post-bg-YesOrNo.jpg
catalog: true
tags:
    - tensorflow
    - summary
---
### tensorflow 1.x 中 batch_norm
写这次总结是因为不止一次的遇到了batch_norm的问题，于是决心记录下来，以免再次绊倒！

#### 使用方法
在构建模型进行分类时发现，模型在测试集和训练集上的结果相差甚远，用训练集测试时结果也很低，当把train_phase 设置
成true时，测试结构明显提高。这明显不符合逻辑，最终定位到batch_norm的使用上。

因为batch_norm 在test的时候，用的是固定的mean和var, 而这个固定的mean和var是通过训练过程中对mean和var进行移动平均得到的。
而直接使用train_op会使得模型没有计算mean和var，因此正确的方式是： 每次训练时应当更新一下moving_mean和moving_var
习惯使用slim包，先看看batch_norm函数
```
@add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               param_initializers=None,
               param_regularizers=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               batch_weights=None,
               fused=None,
               data_format=DATA_FORMAT_NHWC,
               zero_debias_moving_mean=False,
               scope=None,
               renorm=False,
               renorm_clipping=None,
               renorm_decay=0.99,
               adjustment=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"

    Sergey Ioffe, Christian Szegedy

  Can be used as a normalizer function for conv2d and fully_connected. The
  normalization is over all but the last dimension if `data_format` is `NHWC`
  and all but the second dimension if `data_format` is `NCHW`.  In case of a 2D
  tensor this corresponds to the batch dimension, while in case of a 4D tensor
  this
  corresponds to the batch and space dimensions.

  Note: when training, the moving_mean and moving_variance need to be updated.
  By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
  need to be added as a dependency to the `train_op`. For example:

  python
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)

  One can set updates_collections=None to force the updates in place, but that
  can have a speed penalty, especially in distributed settings.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
      Lower `decay` value (recommend trying `decay`=0.9) if model experiences
      reasonably good training performance but poor validation and/or test
      performance. Try zero_debias_moving_mean=True for improved stability.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    param_regularizers: Optional regularizer for beta and gamma.
    updates_collections: Collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    batch_weights: An optional tensor of shape `[batch_size]`,
      containing a frequency weight for each batch item. If present,
      then the batch normalization uses weighted mean and
      variance. (This can be used to correct for bias in training
      example selection.)
    fused: if `None` or `True`, use a faster, fused implementation if possible.
      If `False`, use the system recommended implementation.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
      pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
    scope: Optional scope for `variable_scope`.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_decay: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `decay` is still applied
      to get the means and variances for inference.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example,
        `adjustment = lambda shape: (
          tf.random_uniform(shape[-1:], 0.93, 1.07),
          tf.random_uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
```

 Note: when training, the moving_mean and moving_variance need to be updated.
  By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
  need to be added as a dependency to the `train_op`. For example:

  ```
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)
  ```
  One can set updates_collections=None to force the updates in place, but that
  can have a speed penalty, especially in distributed settings.

在训练时，moving_mean 和 moving_variance 默认是添加到 tf.GraphKeys.UPDATE_OPS 中的，
因此需要作为一个依赖项，在更新train_op时跟新参数。

三种使用方法

1、 
```
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)
```

2、 将 updates_collections参数设置为None，这样会在训练时立即更新，影响速度。
  
3、 使用slim.learning.create_train_op() 创建train_op, 默认将tf.GraphKeys.UPDATE_OPS自动添加为依赖
   和第一种方法相同
   
#### decay参数
decay参数的影响，从公式来看：
```
decay = 0.999 # use numbers closer to1if you have more datatrain_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
```
decay反映了当前估计的衰减速度，decay越小衰减越快（指数衰减），对训练后期的batch mean有更多重视，所以相当于能够更快的热身。
Use a smaller decay value will accelerate the warm-up phase. The default decay is 0.999, for small datasets such like MNIST, you can choose 0.99 or 0.95, and it warms up in a short time.
但是，这建立在训练是可靠的前提下，如果训练本来就跑偏了（loss很大），那么早点热身也没用！正如TF文档中写到的
```
Lower decay value (recommend trying decay=0.9)  if model experiences reasonably good training performance but poor validation and/or test performance. Try zero_debias_moving_mean=True for improved stability.
```
另外，由于使用BN层的网络，预测的时候要用到估计的总体均值和方差，如果iteration还比较少的时候就急着去检验或者预测的话，可能这时EMA估计得到的总体均值/方差还不accurate和stable，
所以会造成训练和预测悬殊，这种情况就是造成下面这个issue的原因：https://github.com/tensorflow/tensorflow/issues/7469
解决的办法就是：当训练结果远好于预测的时候，那么可以通过减小decay，早点“热身”。

默认decay=0.999，一般建议使用0.9

#### 模型保存
当我们使用batch_norm时，slim.batch_norm中的moving_mean和moving_variance不是trainable的，
所以使用saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)无法保存，
应该改为：
```
var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list, max_to_keep=3)
```

#### References
> [1]https://github.com/tensorflow/tensorflow/issues/7469
> [2]https://github.com/tensorflow/tensorflow/issues/1122#issuecomment-280325584
> [3]https://github.com/soloice/mnist-bn
> [4]https://blog.csdn.net/chanbo8205/article/details/86591429


   




