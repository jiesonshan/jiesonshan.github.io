---
layout:     post
title:      tensorflow-Serving部署及调用.md
subtitle:   tensorflow serving
date:       2020-10-17
author:     Jieson
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - tensorflow
    - model deployment
---
### <center>tf-serving 部署模型及调用方法</center>
&#160;&#160;&#160;&#160; Tensorflow Serving 是一个用于机器学习模型 serving 的高性能开源库。它可以将训练好的机器学习模型部署到线上，使用 gRPC 作为接口接受外部调用。更加让人眼前一亮的是，它支持模型热更新与自动模型版本管理。这意味着一旦部署 TensorFlow Serving 后，你再也不需要为线上服务操心，只需要关心你的线下模型训练。
####  环境部署（ubuntu16.04）
tensorflow 2.3

[自己制作docker镜像](https://jiesonshan.github.io/2019/09/12/tf-serving%E9%83%A8%E7%BD%B2%E6%A8%A1%E5%9E%8B/)
#### 部署
本次实验采用docker方式部署
##### cpu
拉取docker镜像
```
docker pull tensorflow/serving:latest
```
本次实验的模型为tf2.3保存的savemodel，
模型目录结构如下：
```
assets/
saved_model.pb
variables/
```
启动服务
```
docker run -t --rm -p 8501:850 -p 8500:8500 \
    -v ~/lab/eye1_encoder:/models/eye1_encoder \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving:latest &
```
参数解释

参数解释：

-p : 端口映射，tensorflow serving提供了两种调用方式：gRPC和REST，
gRPC的默认端口是8500，REST的默认端口是8501.

-v：目录映射，需要注意的是，在新版的docker中，已经移除了–mount type=bind,source=%source_path,target=$target_path的挂载目录方式。

-e：设置变量。

可选参数: MODLE_NAME（默认值：model）

可选参数：MODEL_BASE_PATH（默认值/models）

启动容器之后，相当于在容器中启动服务：
```
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}
```
如果想修改相关配置：
```
docker run --rm -p 8501:850 -p 8500:8500 \
    -v ~/lab/eye1_encoder:/models/eye1_encoder \
    -e MODEL_NAME=eye1_encoder -t --entrypoint=tensorflow_model_server tensorflow/serving:latest --port=8500 --model_name=eye1_encoder &
```
参数解释：

-t --entrypoint=tensorflow_model_server

tensorflow/serving：如果使用稳定版的docker，启动docker之后是不能进入容器内部bash环境的，–entrypoint的作用是允许你“间接”进入容器内部，然后调用tensorflow_model_server命令来启动TensorFlow Serving，这样才能输入后面的参数。
tensorflow serving的详细参数如下：
```
Flags:
    --port=8500                         int32   Port to listen on for gRPC API
    --grpc_socket_path=""               string  If non-empty, listen to a UNIX socket for gRPC API on the given path. Can be either relative or absolute path.
    --rest_api_port=0                   int32   Port to listen on for HTTP/REST API. If set to zero HTTP/REST API will not be exported. This port must be different than the one specified in --port.
    --rest_api_num_threads=16           int32   Number of threads for HTTP/REST API processing. If not set, will be auto set based on number of CPUs.
    --rest_api_timeout_in_ms=30000      int32   Timeout for HTTP/REST API calls.
    --enable_batching=false             bool    enable batching
    --batching_parameters_file=""       string  If non-empty, read an ascii BatchingParameters protobuf from the supplied file name and use the contained values instead of the defaults.
    --model_config_file=""              string  If non-empty, read an ascii ModelServerConfig protobuf from the supplied file name, and serve the models in that file. This config file can be used to specify multiple models to serve and other advanced parameters including non-default version policy. (If used, --model_name, --model_base_path are ignored.)
    --model_name="default"              string  name of model (ignored if --model_config_file flag is set)
    --model_base_path=""                string  path to export (ignored if --model_config_file flag is set, otherwise required)
    --max_num_load_retries=5            int32   maximum number of times it retries loading a model after the first failure, before giving up. If set to 0, a load is attempted only once. Default: 5
    --load_retry_interval_micros=60000000   int64   The interval, in microseconds, between each servable load retry. If set negative, it doesn't wait. Default: 1 minute
    --file_system_poll_wait_seconds=1   int32   Interval in seconds between each poll of the filesystem for new model version. If set to zero poll will be exactly done once and not periodically. Setting this to negative value will disable polling entirely causing ModelServer to indefinitely wait for a new model at startup. Negative values are reserved for testing purposes only.
    --flush_filesystem_caches=true      bool    If true (the default), filesystem caches will be flushed after the initial load of all servables, and after each subsequent individual servable reload (if the number of load threads is 1). This reduces memory consumption of the model server, at the potential cost of cache misses if model files are accessed after servables are loaded.
    --tensorflow_session_parallelism=0  int64   Number of threads to use for running a Tensorflow session. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.
    --tensorflow_intra_op_parallelism=0 int64   Number of threads to use to parallelize the executionof an individual op. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.
    --tensorflow_inter_op_parallelism=0 int64   Controls the number of operators that can be executed simultaneously. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.
    --ssl_config_file=""                string  If non-empty, read an ascii SSLConfig protobuf from the supplied file name and set up a secure gRPC channel
    --platform_config_file=""           string  If non-empty, read an ascii PlatformConfigMap protobuf from the supplied file name, and use that platform config instead of the Tensorflow platform. (If used, --enable_batching is ignored.)
    --per_process_gpu_memory_fraction=0.000000  float   Fraction that each process occupies of the GPU memory space the value is between 0.0 and 1.0 (with 0.0 as the default) If 1.0, the server will allocate all the memory when the server starts, If 0.0, Tensorflow will automatically select a value.
    --saved_model_tags="serve"          string  Comma-separated set of tags corresponding to the meta graph def to load from SavedModel.
    --grpc_channel_arguments=""         string  A comma separated list of arguments to be passed to the grpc server. (e.g. grpc.max_connection_age_ms=2000)
    --enable_model_warmup=true          bool    Enables model warmup, which triggers lazy initializations (such as TF optimizations) at load time, to reduce first request latency.
    --version=false                     bool    Display version
```

显示信息：
```
2020-10-17 02:32:58.633588: I tensorflow_serving/model_servers/server.cc:87] Building single TensorFlow model file config:  model_name: eye1_encoder model_base_path: /models/eye1_encoder
2020-10-17 02:32:58.635587: I tensorflow_serving/model_servers/server_core.cc:464] Adding/updating models.
2020-10-17 02:32:58.635626: I tensorflow_serving/model_servers/server_core.cc:575]  (Re-)adding model: eye1_encoder
2020-10-17 02:32:58.736345: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: eye1_encoder version: 1}
2020-10-17 02:32:58.736402: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: eye1_encoder version: 1}
2020-10-17 02:32:58.736425: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: eye1_encoder version: 1}
2020-10-17 02:32:58.736555: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /models/eye1_encoder/000001
2020-10-17 02:32:58.778136: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2020-10-17 02:32:58.778179: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:234] Reading SavedModel debug info (if present) from: /models/eye1_encoder/000001
2020-10-17 02:32:58.888586: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:199] Restoring SavedModel bundle.
2020-10-17 02:32:59.279835: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:183] Running initialization op on SavedModel bundle at path: /models/eye1_encoder/000001
2020-10-17 02:32:59.367454: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:303] SavedModel load for tags { serve }; Status: success: OK. Took 630905 microseconds.
2020-10-17 02:32:59.377384: I tensorflow_serving/servables/tensorflow/saved_model_warmup_util.cc:59] No warmup data file found at /models/eye1_encoder/000001/assets.extra/tf_serving_warmup_requests
2020-10-17 02:32:59.386745: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: eye1_encoder version: 1}
2020-10-17 02:32:59.393456: I tensorflow_serving/model_servers/server.cc:367] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2020-10-17 02:32:59.396214: I tensorflow_serving/model_servers/server.cc:387] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 238] NET_LOG: Entering the event loop ...
```
##### GPU
前提: 已安装nvidia-docker

拉取gpu版docker镜像
```
docker pull tensorflow/serving:latest-gpu
```
启动服务
```
docker run --runtime=nvidia -t --rm -p 8501:850 -p 8500:8500 \
    -v ~/lab/eye1_encoder:/models/eye1_encoder \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving:latest-gpu &
```
如果要指定GPU，则添加-e NVIDIA_VISIBLE_DEVICES=0 选项
--per_process_gpu_memory_fraction = 0.5 设置GPU可以显存比例
##### 多模型
单一模型部署，上面的方式即可完成。对于多个模型统一部署，基本流程与单模型一致，不同之处在于需要借助模型的配置文件来完成。

model.config

将要部署的模型放在同一个文件夹里，并创建 model.config 文件，文件内容如下：
```
model_config_list {
  config {
    name: 'eye1_encoder',
    model_platform: "tensorflow",
    base_path: '/models/mutil/eye1_encoder'
  },
  config {
    name: 'eye2_encoder',
    model_platform: "tensorflow",
    base_path: '/models/mutil/eye2_encoder'
  }
}
```
启动服务
```
docker run --rm -p 8500:8500 -p 8501:8501 -v /home/jieson/lab/mutil:/models/mutil -t tensorflow/serving:latest --model_config_file=/models/mutil/model.config &
```
结果：
```
2020-10-17 06:07:53.470148: I tensorflow_serving/model_servers/server_core.cc:464] Adding/updating models.
2020-10-17 06:07:53.470195: I tensorflow_serving/model_servers/server_core.cc:575]  (Re-)adding model: eye1_encoder
2020-10-17 06:07:53.470205: I tensorflow_serving/model_servers/server_core.cc:575]  (Re-)adding model: eye2_encoder
2020-10-17 06:07:53.570812: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: eye2_encoder version: 1}
2020-10-17 06:07:53.570876: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: eye2_encoder version: 1}
2020-10-17 06:07:53.570916: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: eye2_encoder version: 1}
2020-10-17 06:07:53.571056: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /models/mutil/eye2_encoder/000001
2020-10-17 06:07:53.613135: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2020-10-17 06:07:53.613178: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:234] Reading SavedModel debug info (if present) from: /models/mutil/eye2_encoder/000001
2020-10-17 06:07:53.670623: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: eye1_encoder version: 1}
2020-10-17 06:07:53.670662: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: eye1_encoder version: 1}
2020-10-17 06:07:53.670685: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: eye1_encoder version: 1}
2020-10-17 06:07:53.670739: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /models/mutil/eye1_encoder/000001
2020-10-17 06:07:53.708971: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2020-10-17 06:07:53.709016: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:234] Reading SavedModel debug info (if present) from: /models/mutil/eye1_encoder/000001
2020-10-17 06:07:53.735904: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:199] Restoring SavedModel bundle.
2020-10-17 06:07:53.811285: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:199] Restoring SavedModel bundle.
2020-10-17 06:07:54.022405: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:183] Running initialization op on SavedModel bundle at path: /models/mutil/eye2_encoder/000001
2020-10-17 06:07:54.084737: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:183] Running initialization op on SavedModel bundle at path: /models/mutil/eye1_encoder/000001
2020-10-17 06:07:54.127839: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:303] SavedModel load for tags { serve }; Status: success: OK. Took 556795 microseconds.
2020-10-17 06:07:54.137134: I tensorflow_serving/servables/tensorflow/saved_model_warmup_util.cc:59] No warmup data file found at /models/mutil/eye2_encoder/000001/assets.extra/tf_serving_warmup_requests
2020-10-17 06:07:54.147400: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: eye2_encoder version: 1}
2020-10-17 06:07:54.179978: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:303] SavedModel load for tags { serve }; Status: success: OK. Took 509246 microseconds.
2020-10-17 06:07:54.189399: I tensorflow_serving/servables/tensorflow/saved_model_warmup_util.cc:59] No warmup data file found at /models/mutil/eye1_encoder/000001/assets.extra/tf_serving_warmup_requests
2020-10-17 06:07:54.199335: I tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: eye1_encoder version: 1}
2020-10-17 06:07:54.201169: I tensorflow_serving/model_servers/server.cc:367] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
[evhttp_server.cc : 238] NET_LOG: Entering the event loop ...
2020-10-17 06:07:54.202394: I tensorflow_serving/model_servers/server.cc:387] Exporting HTTP/REST API at:localhost:8501 ...
```
调用方式单模型一样

##### 版本控制
tensorflow serving服务默认只会读取最大版本号的版本（按数字来标记版本号），实际上，我们可以通过提供不同的版本的模型，比如提供稳定版、测试版，来实现版本控制，只需要在在配置文件中配置model_version_policy。
```
model_config_list {
  config {
    name: 'eye1_encoder',
    model_platform: "tensorflow",
    base_path: '/models/mutil/eye1_encoder'
    model_version_policy{
      specific{
            version: 1,
            version: 2
        }
    }
    version_labels{
        key: "stable",
        value: 1
    }
    version_labels{
        key: "test",
        value: 2
    }
  },
  config {
    name: 'eye2_encoder',
    model_platform: "tensorflow",
    base_path: '/models/mutil/eye2_encoder'
  }
}
```
提供版本号为1和版本号为2的版本，并分别为其取别名stable和test。这样做的好处在于，用户只需要定向到stable或者test版本，而不必关心具体的某个版本号，同时，在不通知用户的情况下，可以调整版本号，比如版本2稳定后，可以升级为稳定版，只需要将stable对应的value改为2即可。同样，若需要版本回滚，将value修改为之前的1即可。

启动方式如上。

#### 客户端
tensorflow serving提供两种调用方式，REST 和 gPRC方式，REST方式比较简单
##### REST
```
url = 'http://localhost:8501/v1/models/eye1_encoder:predict'
data = np.random.random((1, 128, 128, 1))

params = {'inputs': data.tolist()}
t = time.time()
res = requests.post(url, json=params)
print(time.time() - t)
print(res)
result = res.json()
img = np.array(result['outputs'])
print(img.shape)
print(result)
```
##### gPRC
安装tensorflow-serving-api
```
pip install tensorflow-serving-api
```
查看模型结构
```
saved_model_cli show --dir=. --all
```
模型结构信息
```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 128, 128, 1)
        name: serving_default_input_1:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_1'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 512)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          x: TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name='x')
        Argument #2
          DType: bool
          Value: False
    Option #2
      Callable with:
        Argument #1
          x: TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name='x')
        Argument #2
          DType: bool
          Value: True
    Option #3
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name='input_1')
        Argument #2
          DType: bool
          Value: False
    Option #4
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name='input_1')
        Argument #2
          DType: bool
          Value: True

  Function Name: '_default_save_signature'
    Option #1
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name='input_1')

  Function Name: 'call_and_return_all_conditional_losses'
    Option #1
      Callable with:
        Argument #1
          x: TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name='x')
        Argument #2
          DType: bool
          Value: False
    Option #2
      Callable with:
        Argument #1
          x: TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name='x')
        Argument #2
          DType: bool
          Value: True
    Option #3
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name='input_1')
        Argument #2
          DType: bool
          Value: True
    Option #4
      Callable with:
        Argument #1
          input_1: TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32, name='input_1')
        Argument #2
          DType: bool
          Value: False
```
注意：请牢记上面的signature_def值和inputs值及其shape，在grpc调用的时候需要用到

定义请求体
```
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc


def request_server(img_resized, server_url):
    '''
    for tensorflow serving seed request
    :param img_resized: processed image for inference. numpy array, shape:(h, w, 3)
    :param server_url: Tensorflow serving
    :return: model predict result numpy array
    '''
    # Request
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'eye1_encoder'  # model name start docker model name param
    request.model_spec.signature_name = 'serving_default'
    request.inputs["input_1"].CopyFrom(
        tf.make_tensor_proto(img_resized, shape=[1, ]+list(img_resized.shape))
    )
    response = stub.Predict(request, 5.0)
    return np.asarray(response.outputs['output_1'].float_val)  # model output node name


data = np.random.random((128, 128, 1))
data = data.astype(np.float32)
server_url = '0.0.0.0:8500'
t = time.time()
out = request_server(data, server_url)
print(time.time() - t)
print(out.shape)
print(out)
```
#### reference
> [1] https://blog.csdn.net/chenguangchun1993/article/details/104971811/
> [2] https://www.tensorflow.org/tfx/serving/serving_basic?hl=zh_cn
> [3] https://blog.csdn.net/zong596568821xp/article/details/99715005
> [4] https://zhuanlan.zhihu.com/p/42905085
> [5] https://blog.csdn.net/u013714645/article/details/81449487
> [6] https://zhuanlan.zhihu.com/p/96917543