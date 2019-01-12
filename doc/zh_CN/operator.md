# 算符支持

----

这里列举出，框架已经支持的算符，以及对应的参数设置。

所有内置算符都包含参数：
- `#op`: `string` 算符的名称
- `#name`: `string` 实例化名称
- `#output_count`: `int` 输出结果数


## 内置算符

### _field(a) -> field
描述：用于获取 `Packed Tensor` 的元素。  
输入：`a` `PackedTensor` 压缩数据格式  
输出：`field` `Tensor`  
参数：
- `offset`: `uint` `[Required]` 要取输入的 `Packed Tensor` 的偏移下标

### _pack(a...) -> packed
描述：打包输入元素，输出 `Packed Tensor`  
输入：`a` `List<Tensor>` 要进行打包的数据  
输出：`packed` `PackedTensor`  
参数：  

### _resize2d(a, size) -> resized
描述：对输入的原图和要缩放的大小，输出缩放后的图像  
输入：`a` `Tensor` 要缩放的图像  
输入：`size` `IntArray1D` 数组的长度，要和 `a` 的维度一致。要缩放的图像的大小，包含`-1`表示扩展  
输出：`resized` `Tensor` 缩放后的图像   
参数：
- `type`: `Enum[linear=0, cubic=1] Default(0)` `[Optional]` 图像缩放类型  

举例：  
输入: `[1, 640, 480, 3]` 和 `[-1, 300, 300, -1]`，输出 `[1, 300, 300, 3]`。

说明：  
`size`，中`-1`表示通道维度，例如`[-1, 300, 300]`,表示要将输入的`CHW`格式图像缩放到`(300, 300)`大小。
其中，不为`-1`的数值必须大于`0`，存在且只存在两个，两个必须在相邻维度。
例如：`[-1, -1, 3]` 和 `[400, -1, 300]` 都是错误输入。

### _image_resize2d(a, size) -> resized
描述：对输入的原图和要缩放的大小，输出缩放后的图像  
输入：`a` `Tensor` 要缩放的图像  
输入：`size` `IntArray1D` 数组的长度，要和 `a` 的维度一致。要缩放的图像的大小，包含`-1`表示扩展  
输出：`resized` `Tensor` 缩放后的图像   
参数：
- `type`: `Enum[linear=0, cubic=1] Default(0)` `[Optional]` 图像缩放类型  

举例：  
输入: `[1, 640, 480, 3]` 和 `[-1, 300, 300, -1]`，输出 `[1, 300, 300, 3]`。

说明：  
该算符和 `_resize2d` 含义相同，但是限制输入的tensor维度为4或以下。

### _transpose(a) -> b
描述：对输入的 Tensor 进行维度转换，输出转换后的图像  
别名：`permute`  
输入：`a` `Tensor` 要转换的数据  
输出：`b` 转换后的数据  
参数：   
- `permute`: `IntArray1D` `[Optional]` 数组的长度，要和 `a` 的维度一致。输入的第 `i` 个维度就是 `permute[i]` 的维度。
如果，没有设置该参数，则相当于普通矩阵转置。

举例：  
如果 `a` 的 `shape` 为 `[1, 640, 480, 3]`，`permute` 为 `[0, 3, 1, 2]`，
输出 `b` 的 `shape` 为 `[1, 3, 640, 480]`。数据类型不变。

说明：  
`permute` 中的元素必须是有效维度下标，且每个下标有且必须出现一次。`tranpose` 会对存储格式产生影响。

### _reshape(a) -> b
描述：对输入的 Tensor 进行维度变换，输出转换后的数据  
输入：`a` `Tensor` 要转换的数据  
输出：`b` 转换后的数据  
参数：   
- `shape`: `IntArray1D` `[Required]` 输出的 `shape` 要和此参数一致，中间可以出现最多一个 `-1` 表示维度填充，保证输出的元素个数和输入的元素个数一致。

举例：  
如果 `a` 的 `shape` 为 `[4, 2700]`，`shape` 为 `[-1, 300, 300, 3]`，
输出 `b` 的 `shape` 为 `[4, 300, 300, 3]`。数据类型不变。

说明：  
此操作只影响 `Tensor` 的 `shape` 不会对内存布局产生影响。

### _dimshuffle(x) -> y
描述：对输入的 Tensor 进行 channel 变换，输出转换后的数据  
输入：`x` `Tensor` 要转换的数据    
输出：`y` `Tensor`  

参数：  
- `dim` `int` 要进行 `shuffle` 的维度  
- `shuffle` `IntArray` 维度 `>= 1`，每个元素必须属于 `[0，rank(x, dim)]`

举例：  
如果 `x` 的 `shape` 为 `[300, 300, 3]` 那么 `_dimshuffle(x, 2, [2, 1, 0])`
完成了原始数据的第`2`个维度进行了按照 `shuffle` 进行了 shuffle。输出仍旧是 `[300, 300, 3]`。       

说明：  
`shuffle` 的维度可以比对应 `x` 的 `dim` 维度多或者少。从而完成通道复制的效果。
如果 `x` 的 `shape` 为 `[100, 100, 1]` 那么 `_dimshuffle(x, 2, [0, 0, 0])`
的输出 `shape` 为 `[100, 100, 3]`

### conv2d(x, w) -> y
描述：对输入的 Tensor 进行 二维卷积操作，输出卷积后的数据
输入：`x` 输入数据
输入：`w` `Tensor4D` 格式为 `[output_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor`

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding` `IntArray4Dx2D`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `IntArray4D`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dialations` `IntArray4D`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
`type` 在当前版本中，固定为 `NCHW`。
输出大小计算除法时，向下取整，最小为1。默认0padding。  
输出大小的计算公式为：  
```
pad_h = pad_h_top + pad_h_bottom
pad_w = pad_w_left + pad_h_right
output_h = floor((height + pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1);
output_w = floor((width + pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1);
```

### _shape(x) -> shape
描述：对输入的 `Tensor` 返回对应的 `shape`。  

说明：  
返回的 `shape` 默认在 `CPU` 存储上。

### pad(x, padding) -> y
描述：对输入的 `Tensor` 进行 `pad`。

参数：  
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数

说明：  
其中 `padding` 的第一个维度和 `x` 的 `shape` 维度一致，第二个维度为常数 `2`。
输出的大小按照 `padding` 后的大小计算，最小为 `1`。


### depthwise_conv2d(x, w) -> y
描述：对输入的 Tensor 进行`Depthwise`二维卷积操作，输出卷积后的数据
输入：`x` 输入数据
输入：`w` `Tensor4D` 格式为 `[multiplier_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor` 输出的 channel 数量为 `multiplier_channels * input_channels`。

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding` `IntArray4Dx2D`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `IntArray4D`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dialations` `IntArray4D`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
输出大小计算除法时，向下取整，最小为1。默认0padding。


### add_bias(x, b)
描述：对输入的 Tensor 加上偏置。  
输入：`x` `Tensor` 输入数据  
输入：`b` `Array1D` 维度和通道数相同，通道的维度通过 `dim` 来指定。  
输出：`y` `Tensor`  

参数：  
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `dim` `IntScalar` 通道所在的维度

说明：
`format` 和 `dim` 至少设置一项即可。


### padding_conv2d(x, padding, w) -> y
描述：等价于 `conv2d(pad(padding), w)`

参数：  
包含 `conv2d` 和 `padding` 参数

### conv2d_bias(x, w, b) -> y
描述：等价于 `bias(conv2d(x, w), b)`

参数：
包含 `conv2d` 和 `bias` 参数。

### padding_conv2d_bias(x, padding, w, b) -> y
描述：等价于 `bias(conv2d(pad(x, padding), w), b)`

参数：
包含 `pad`、`conv2d` 和 `bias` 参数。

### batch_norm(x, mean, variance) -> y
描述：单纯进行 BN

参数：
- `dim`: `Int` 表示`channel`所在的维度
- `epsilon`: `Scalar Default(0.001)` 表示约束系数，作用见说明

说明：
`y = (x - mean) / (sqrt(var + epsilon))`

### batch_scale(x, scale, bias)
描述：单纯进行 Scale

参数：
- `dim`: `Int` 表示`channel`所在的维度

说明：
`y = x * scale + bias`

### fused_batch_norm(x, gamma, beta, mean, variance) -> y
描述：等价于 `batch_scale(batch_norm(mean, variance), gamma, beta)`

参数：  
包含 `batch_scale`、`batch_norm` 参数。

### add(x, a) -> y
描述：进行矩阵加法，支持 `Broadcast`

参数：  
无

说明：
`y_i = x_i + a_i`，要求`a`和`x`的维度一样，或者为`1`。 
关于广播的含义见附1。

### sub(x, a) -> y
描述：进行矩阵加法，支持 `Broadcast`

参数：  
无

说明：
`y_i = x_i - a_i`，要求`a`和`x`的维度一样，或者为`1`。  
关于广播的含义见附1。

### mul(x, a) -> y
描述：进行矩阵加法，支持 `Broadcast`

参数：  
无

说明：
`y_i = x_i * a_i`，要求`a`和`x`的维度一样，或者为`1`。 
关于广播的含义见附1。

### div(x, a) -> y
描述：进行矩阵加法，支持 `Broadcast`

参数：  
无

说明：
`y_i = x_i / a_i`，要求`a`和`x`的维度一样，或者为`1`。 
关于广播的含义见附1。

### inner_prod(x, a) -> y
描述：`y = x \mul a`

### relu(x) -> y
描述： `y = x > 0 ? x : 0`

### relu_max(x) -> y
描述： `y = min(x > 0 ? x : 0, max)`

参数：
- `max`: `Scalar` 输出的最大值。

### sigmoid(x) -> y
描述： `y = 1 / (1 + exp(-x))`

### prelu
描述：`y = x > 0 ? x : slope * x`

参数：
- `dim`: `Int` slope 所在的维度，当slope维度不为1时，此参数必须设置。
- `slope` `ArrayND` 或者 `Scalar` 维度为1或者与`dim`给定的维度相同

### softmax
描述：

参数：  
- `dim`: `Int` softmax 要处理的维度 

### concat
描述：链接算符

参数：
- `dim`: 要进行Concat的维度。

### pooling2d(x)
描述：进行下采样

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `type` `Enum[max=0, avg=1]` 
- `padding` `IntArray4Dx2D`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_type` `Enum[black=0, copy=1, loop=2] Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `ksize`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
计算大小时的除法结果，向上取整，最小为1。

pooling size计算公式:
```c
output_h = ceil((input_h + pad_h_up + pad_h_down - kernel_h) / static_cast<float>(stride_h) + 1);
output_w = ceil((input_w + pad_w_left + pad_w_right - kernel_w) / static_cast<float>(stride_w) + 1);
```
padding_type为black时，超出可计算区域的结果为0。

### pooling2d_v2(x, padding, ksize, stride)
描述：进行下采样
- `padding` `IntArray4Dx2D`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `ksize`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

参数：

- `format` `String` 为 `NCHW` 或者 `NHWC`
- `type` `Enum[max=0, avg=1]` 
- `padding_type` `Enum[black=0, copy=1, loop=2] Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
计算大小时的除法结果，向上取整，最小为1。
padding_type为black时，超出可计算区域的结果为0。

### padding_pooling2d_v2(x, padding, ksize, stride)
描述：

参数：  
- `padding_type` `Enum[black=0, copy=1, loop=2] Default(0)` `[Optional]` 表示 `padding` 时填充的参数

### flatten(x) -> y
描述：把输入的shape，调整成2维的矩阵。

参数：  
无

说明：  
输入 `x` 的 `shape` 为 `[1, 20, 3, 3]`，输出的 `shape` 为 `[1, 180]`。

## to_float(x) -> y
描述：把输入类型，调整成 `float` 类型。

### tf_conv2d_padding

### tf_pooling2d_padding

### mx_conv2d_padding

### mx_pooling2d_padding

## 附录

1. 在做基础运算的时候，`x`和`a`有会三种意义，分别为`标量`，`张量`和`广播张量`。这里的广播张量的意义为：
   加入 x 的 shape 为 `[10, 10]` `a` 的 `shape` 为 `[10, 1]`，这个时候
   `y_{ij} = x_{ij} + a_{i0}` 在 `a` 的第二个维度，实现了广播。  
   注意： 广播的维度可以在矩阵中存在多份，默认维度大小为 `1` 的都支持广播。
   
2. 框架支持的 `padding` 类型。  
   tf_valid:
   ```
   output_height = ceil((input_height + 2 * m_pad_h - m_kernel_h + 1) / (float)m_stride_h);
   output_width = ceil((input_width + 2 * m_pad_w - m_kernel_w + 1) / (float)m_stride_w);
   ```
   tf_same:
   ```
   output_height = ceil((input_height + 2 * m_pad_h) / (float)m_stride_h);
   output_width = ceil((input_width + 2 * m_pad_w) / (float)m_stride_w);
   ```
   caffe:
   ```
   output_height = ceil((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
   output_width = ceil((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
   ```
   mx_valid:
   ```
   output_height = floor((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
   output_width = floor((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
   ```

