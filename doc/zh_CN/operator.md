# 算符支持

----

这里列举出，框架已经支持的算符，以及对应的参数设置。

所有内置算符都包含参数：
- `#op`: `String` 算符的名称
- `#name`: `String` 实例化名称
- `#output_count`: `Int` 输出结果数
- `#shape`: `IntArray` 大小


## 内置算符

### _field(a) -> field
描述：用于获取 `Packed Tensor` 的元素。  
输入：`a`: `Tensor` 压缩数据格式  
输出：`field` `Tensor`  
参数：
- `offset`: `Int` `[Required]` 要取输入的 `Packed Tensor` 的偏移下标

### _pack(a...) -> packed
描述：打包输入元素，输出 `Packed Tensor`  
输入：`a`: `List<Tensor>` 要进行打包的数据  
输出：`packed` `Tensor`  
参数：无

### _resize2d(x..device, size..host) -> y..device
描述：对输入的原图和要缩放的大小，输出缩放后的数据  
输入：`x`: `Tensor` 要缩放的数据  
输入：`size`: `IntArray` 数组的长度，要和 `a` 的维度一致。要缩放的图像的大小，包含`-1`表示扩展  
输出：`y`: `Tensor` 缩放后的图像   
参数：
- `type`: `Enum[linear=0, cubic=1] Default linear` `[Optional]` 图像缩放类型  

举例：  
输入: `Tensor[1, 640, 480, 3]` 和 `[-1, 300, 300, -1]`，输出 `Tensor[1, 300, 300, 3]`。

说明：  
`$x.shape.size == $size.size`  
`size`，中`-1`表示通道维度，例如`[-1, 300, 300]`,表示要将输入的`CHW`格式图像缩放到`(300, 300)`大小。
其中，不为`-1`的数值必须大于`0`，存在且只存在两个，两个必须在相邻维度。
例如：`size` 为 `[-1, -1, 3]` 和 `[400, -1, 300]` 都是错误输入。

### _transpose(x..device) -> y.device
描述：对输入的 Tensor 进行维度转换，输出转换后的图像  
别名：`permute`  
输入：`x`: `Tensor` 要转换的数据  
输出：`y`: `Tensor` 转换后的数据  
参数：   
- `permute`: `IntArray` `[Optional]` 数组的长度，要和 `a` 的维度一致。输入的第 `i` 个维度就是 `permute[i]` 的维度。
如果，没有设置该参数，则相当于普通矩阵转置。

举例：  
如果 `x.shape` 为 `[1, 640, 480, 3]`，`permute` 为 `[0, 3, 1, 2]`，
输出 `b.shape` 为 `[1, 3, 640, 480]`。数据类型不变。

说明：  
`permute` 中的元素必须是有效维度下标，且每个下标有且必须出现一次。`tranpose` 会对存储格式产生影响。

### _reshape(x..device) -> y.device
描述：对输入的 Tensor 进行维度变换，输出转换后的数据  
输入：`x` `Tensor` 要转换的数据  
输出：`y` 转换后的数据  
参数：   
- `shape`: `IntArray` `[Required]` 输出的 `shape` 要和此参数一致，中间可以出现最多一个 `-1` 表示维度填充，保证输出的元素个数和输入的元素个数一致。

举例：  
如果 `x.shape` 为 `[4, 2700]`，`shape` 为 `[-1, 300, 300, 3]`，
输出 `y.shape` 为 `[4, 300, 300, 3]`。数据类型不变。

说明：  
此操作只影响 `Tensor` 的 `shape` 不会对内存布局产生影响。

### _dimshuffle(x..device) -> y..device
描述：对输入的 Tensor 进行 channel 变换，输出转换后的数据  
输入：`x` `Tensor` 要转换的数据    
输出：`y` `Tensor`  

参数：  
- `dim` `Int` 要进行 `shuffle` 的维度  
- `shuffle` `IntArray` 维度 `>= 1`，每个元素必须属于 `[0，rank(x, dim)]`

举例：  
如果 `x.shape` 为 `[300, 300, 3]` 那么 `_dimshuffle(x, 2, [2, 1, 0])`
完成了原始数据的第`2`个维度进行了按照 `shuffle` 进行了 shuffle。输出仍旧是 `[300, 300, 3]`。       

说明：  
`shuffle` 的维度可以比对应 `x` 的 `dim` 维度多或者少。从而完成通道复制的效果。
如果 `x` 的 `shape` 为 `[100, 100, 1]` 那么 `_dimshuffle(x, 2, [0, 0, 0])`
的输出 `shape` 为 `[100, 100, 3]`

### conv2d(x..device, w..device) -> y..device
描述：对输入的 Tensor 进行 二维卷积操作，输出卷积后的数据
输入：`x` `Tensor4D` 输入数据
输入：`w` `Tensor4D` `shape` 为 `[output_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor4D`

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
`type` 在当前版本中，固定为 `NCHW`。
输出大小计算除法时，向下取整，最小为`1`。默认`0` `padding`。  
输出大小的计算公式为：  
```
pad_h = pad_h_top + pad_h_bottom
pad_w = pad_w_left + pad_h_right
output_h = floor((height + pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1);
output_w = floor((width + pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1);
```

### _shape(x..device) -> shape..host
描述：对输入的 `Tensor` 返回对应的 `shape`。  

说明：  
返回的 `shape` 默认在 `CPU` 存储上。

### pad(x..device, padding..host) -> y..device
描述：对输入的 `Tensor` 进行 `pad`。

参数：  
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数

说明：  
其中 `padding` 的第一个维度和 `x` 的 `shape` 维度一致，第二个维度为常数 `2`。
输出的大小按照 `padding` 后的大小计算，最小为 `1`。


### depthwise_conv2d(x, w) -> y
描述：对输入的 Tensor 进行`Depthwise`二维卷积操作，输出卷积后的数据
输入：`x` `Tensor4D` 输入数据
输入：`w` `Tensor4D` 格式为 `[multiplier_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor4D` 输出的 channel 数量为 `multiplier_channels * input_channels`。

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
输出大小计算除法时，向下取整，最小为1。默认0padding。


### add_bias(x..device, b..device)
描述：对输入的 Tensor 加上偏置。  
输入：`x` `Tensor` 输入数据  
输入：`b` `Array` 维度和通道数相同，通道的维度通过 `dim` 来指定。  
输出：`y` `Tensor`  

参数：  
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `dim` `Int` 通道所在的维度

说明：
`format` 和 `dim` 至少设置一项即可。


### conv2d_v2(x..device, padding..host, w..device) -> y..device
描述：对输入的 Tensor 进行 二维卷积操作，输出卷积后的数据
输入：`x` `Tensor4D` 输入数据  
输入：`padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
   在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
   在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
输入：`w` `Tensor4D` 格式为 `[output_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor4D`

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
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


### depthwise_conv2d_v2(x..device, padding..host, w..device) -> y..device
描述：对输入的 Tensor 进行`Depthwise`二维卷积操作，输出卷积后的数据
输入：`x` `Tensor4D` 输入数据  
输入：`padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
   在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
   在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
输入：`w` `Tensor4D` 格式为 `[multiplier_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor4D` 输出的 channel 数量为 `multiplier_channels * input_channels`。

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
输出大小计算除法时，向下取整，最小为1。默认0padding。

### batch_norm(x..device, mean..device, variance..device) -> y..device
描述：单纯进行 BN
输入：`x`: `Tensor4D`  
输入：`mean`: `Array` `$mean.size == $x.shape.size`  
输入：`variance`: `Array` `$mean.size == $x.shape.size`   
输出：`y`: `Tensor4D` `$y.shape == $x.shape` 

参数：
- `dim`: `Int` 表示`channel`所在的维度
- `epsilon`: `Scalar Default(10e-5)` 表示约束系数，作用见说明

说明：
`y = (x - mean) / (sqrt(var + epsilon))`

### batch_scale(x..device, scale..device, bias..device)
描述：单纯进行 Scale
输入：`x`: `Tensor4D`  
输入：`mean`: `Array` `$mean.size == $x.shape.size`  
输入：`variance`: `Array` `$mean.size == $x.shape.size`    
输出：`y`: `Tensor4D` `$y.shape == $x.shape`

参数：
- `dim`: `Int` 表示`channel`所在的维度

说明：
`y = x * scale + bias`

### fused_batch_norm(x, gamma, beta, mean, variance) -> y
描述：等价于 `batch_scale(batch_norm(mean, variance), gamma, beta)`  
输入：`x`: `Tensor4D`  
输入：`mean`: `Array` `$mean.size == $x.shape.size`  
输入：`variance`: `Array` `$mean.size == $x.shape.size`  
输入：`mean`: `Array` `$mean.size == $x.shape.size`  
输入：`variance`: `Array` `$mean.size == $x.shape.size`  
输出：`y`: `Tensor4D` `$y.shape == $x.shape`

参数：  
包含 `batch_scale`、`batch_norm` 参数。

### add(x..device, a..device) -> y..device
描述：进行矩阵加法，支持 `Broadcast`  
输入: `x`: `Tensor`  
输入: `a`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
无

说明：
`y_i = x_i + a_i`，要求`a`和`x`的维度一样，或者为`1`。 
关于广播的含义见附1。

### sub(x..device, a..device) -> y..device
描述：进行矩阵加法，支持 `Broadcast`  
输入: `x`: `Tensor`  
输入: `a`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
无

说明：
`y_i = x_i - a_i`，要求`a`和`x`的维度一样，或者为`1`。  
关于广播的含义见附1。

### mul(x..device, a..device) -> y..device
描述：进行矩阵加法，支持 `Broadcast`  
输入: `x`: `Tensor`  
输入: `a`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
无

说明：
`y_i = x_i * a_i`，要求`a`和`x`的维度一样，或者为`1`。 
关于广播的含义见附1。

### div(x..device, a..device) -> y..device
描述：进行矩阵加法，支持 `Broadcast`  
输入: `x`: `Tensor`  
输入: `a`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
无

说明：
`y_i = x_i / a_i`，要求`a`和`x`的维度一样，或者为`1`。 
关于广播的含义见附1。

### inner_prod(x..device, a..device) -> y..device
描述：`y = x \mul a`  
输入: `x`: `Matrix`  
输入: `a`: `Matrix`  
输出: `y`: `Matrix`  

### relu(x) -> y
描述： `y = x > 0 ? x : 0`  
输入: `x`: `Tensor`  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

### relu_max(x) -> y
描述： `y = min(x > 0 ? x : 0, max)`  
输入: `x`: `Tensor`  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

参数：
- `max`: `Scalar` 输出的最大值。

### sigmoid(x) -> y
描述： `y = 1 / (1 + exp(-x))`  
输入: `x`: `Tensor`  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

### prelu(x..device, slope..device) -> y
描述：`y = x > 0 ? x : slope * x`  
输入: `x`: `Tensor`  
输入: `slope`: `Array` 维度与`dim`给定的维度相同  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

参数：
- `dim`: `Int` slope 所在的维度，此参数必须设置。


说明：  
`$slope.size == $x.shape($dim)`

### softmax(x) -> y
描述：  
输入: `x`: `Tensor`  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

参数：  
- `dim`: `Int` softmax 要处理的维度 
- `smooth`: `Int` 非零表示真，0表示假

说明：  
smooth 为0时：
```
y_i = exp(-x_i) / \sum{exp(-x_i)}
```
smooth 为非0时：
```
t_i = x_i - max(x)
y_i = exp(-t_i) / \sum{exp(-t_i)}
```


### concat(x...) -> y
描述：链接算符  
输入: `x`: `List<Tensor>` 要进行拼接的数据  
输出: `y`: `Tensor`  

参数：
- `dim`: `Int` 要进行Concat的维度。

说明：  
要把输的元素进行拼接，在`dim`维度上，除了`dim`维度，其他的输入的数据的维度必须相同。
输出的`dim`维度是输入对应`dim`维度的和。

### pooling2d(x)
描述：进行下采样

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `type` `Enum[max=0, avg=1]` 
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_type` `Enum[black=0, copy=1, loop=2] Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `ksize` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
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
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `ksize` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

参数：

- `format` `String` 为 `NCHW` 或者 `NHWC`
- `type` `Enum[max=0, avg=1]` 
- `padding_type` `Enum[black=0, copy=1, loop=2] Default black` `[Optional]` 表示 `padding` 时填充的参数

说明：  
计算大小时的除法结果，向上取整，最小为1。
padding_type为black时，超出可计算区域的结果为0。

### flatten(x) -> y
描述：把输入的shape，调整成2维的矩阵。  
输入: `x`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
- `dim`: `Int` `[Optional] Default=1`

说明：  
输入 `x` 的 `shape` 为 `[1, 20, 3, 3]`，输出的 `shape` 为 `[1, 180]`。
在对应 `dim` 位置进行拉伸。
输入 `x` 的 `shape` 为 `[1, 20, 3, 3]`，`dim = 2`，输出的 `shape` 为 `[1, 20, 9]`。
输入 `x` 的 `shape` 为 `[2, 3]`，`dim = 2`，输出的 `shape` 为 `[2, 3, 1]`。

## to_float(x) -> y
描述：把输入类型，调整成 `float` 类型。  
输入: `x`: `Tensor`  
输出: `y`: `Tensor`  

## prewhiten(x) -> y
描述：进行图像白化  
输入: `x`: `Tensor`  
输出: `y`: `Tensor`  

说明：  
对于输入的每一个样本 `x_i` 执行：
```cpp
template <typename T>
void prewhiten(T *data, size_t len)
{
	double mean = 0;
	double std_dev = 0;
	T *at= nullptr;

	at = data;
	for (size_t i = 0; i < len; ++i, ++at) mean += *at;
	mean /= len;

	at = data;
	for (size_t i = 0; i < len; ++i, ++at) std_dev += (*at - mean) * (*at - mean);
	std_dev = std::sqrt(std_dev / len);
	std_dev = std::max<T>(std_dev, 1 / std::sqrt(len));
	double std_dev_rec = 1 / std_dev;

	at = data;
	for (size_t i = 0; i < len; ++i, ++at) {
		*at -= mean;
		*at *= std_dev_rec;
	}
}
```

### _cast(x..device, ) -> y
输入：`x`   

参数：  
- `dtype` `Int` 要转换成的类型

### gather(x..device, indices..host) -> y

参数：  
- `axis` `Int` 要进行gather的维度。

说明：  
等价于`numpy.take(x, indices, axis=axis)`

### unsqueeze(x..device) -> y

参数：  
- `axes` `IntArray` 要填充维度

说明：  
等价于`numpy.expend_dims(x, axis) for axis in axes`


### _reshape_v2(x..device, shape..host) -> y.device
描述：对输入的 Tensor 进行维度变换，输出转换后的数据  
输入：`x` `Tensor` 要转换的数据  
输入：`shape`: `IntArray` 输出的 `shape` 要和此参数一致，中间可以出现最多一个 `-1` 表示维度填充，保证输出的元素个数和输入的元素个数一致。  
输出：`y` 转换后的数据  

举例：  
如果 `x.shape` 为 `[4, 2700]`，`shape` 为 `[-1, 300, 300, 3]`，
输出 `y.shape` 为 `[4, 300, 300, 3]`。数据类型不变。

说明：  
此操作只影响 `Tensor` 的 `shape` 不会对内存布局产生影响。


### gemm(a..device, b..device, c..device) -> Y..device
描述：就是GEMM，嗯。  
输入：`A` `Matrix`
输入：`B` `Matrix`
输入：`C` `Matrix` 或者可以广播的 `Tensor` 
输入：`Y` `Matrix`

参数：
- `alpha` `Float`
- `beta` `Float`
- `transA` `Int` 布尔变量
- `transB` `Int` 布尔变量

说明：
```
A' = transpose(A) if transA else A
B' = transpose(B) if transB else B
Compute Y = alpha * A' * B' + beta * C,
where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K),
input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N).
A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
This operator supports unidirectional broadcasting
(tensor C should be unidirectional broadcastable
to tensor A * B); 
```

### lrn (x..device) -> y..device = delete

参数：
- `dim` `Int` 要进行LRN的维度
- `alpha` `Float` `Default(0.0001)`
- `beta` `Float` `Default(0.75)`
- `bias` `Float` `Default(1)`
- `size` `Int` `Required`

说明：  
按照 `LRN` 的传统公式和做法

### global_pooling2d(x)
描述：进行全局下采样

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `type` `Enum[max=0, avg=1]` 

说明：  
输出大小固定为1x1。

### _limit(x..device) -> y..device
描述：对blob大小进行限制，如果输入大于这个大小，则进行center_crop，
否则保持原有大小。

参数：  
- `shape`: `IntArray` 输出限制，维度要小于x的维度。-1表示不进行限制。

说明：  
假如shape小于x的大小，则在shape高位补充-1，直到维度相同，再进行调整。

### shape_index_patch(x..device, pos..device) -> y..device
描述：根据pos在x上进行采样。  
输入：`x`: `Tensor4D` shape 为 `[number, channels, height, width]`  
输入：`pos`: `Tensor4D` shape 为 `[number, landmark, 1, 1]`  

输出：`y`: `Tensor5D` shape 为 `[number, channels, x_patch_h, landmark / 2, x_patch_w]`
其中 `x_patch_h = int(origin_patch.h * x.height / origin.h + 0.5)`,  
`x_patch_w = int(origin_patch.w * x.width / origin.w + 0.5)`,  
Note: 这是对应某一个实现的版本。


参数：  
- `origin_patch`: `Int[2]{h, w}`  
- `origin`: `Int[2]{h, w}`  

说明：  
`pos.number == x.number`，根据pos表示的位置信息，在对应位置crop出`[x_patch_h, x_patch_w]`大小。

### _nhwc_resize2d(x..device) = delete

参数：  
- `size` `Int[2]` 内容为 `{width, height}`。

### _nhwc_crop2d(x..device) = delete

参数：  
- `rect` `Int[4]` 内容为 `{x, y, width, height}`。

### _nhwc_center_crop2d(x..device)

参数：  
- `size` `Int[2]` 内容为 `{width, height}`。

### _nhwc_channel_swap(x..device) = delete

参数：  
- `shuffle` `IntArray`

### _nhwc2nchw(x..device) = delete

参数：无

### _tf_conv2d_padding(x..device, w..device) -> dynamic_padding

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding_method` `String` `[Required]` 表示 `padding` 方式为`SAME` 或 `VALID`
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding` `Int[4, 2]` `[Optional]` `[Default] Zero padding` 静态进行padding的数据
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

### _tf_pooling2d_padding（x, ksize, stride) -> dynamic_padding
描述：  
- `x` `Tensor4D` 预计要进行 padding 的数据
- `ksize` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dynamic_padding`  `Int[4, 2]`输出的padding形式，为4x2维

参数：  
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding_method` `String` `[Required]` 表示 `padding` 方式为`SAME` 或 `VALID`
- `padding` `Int[4, 2]` `[Optional]` `[Default] Zero padding` 静态进行padding的数据
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。


### mx_conv2d_padding = delete

### _mx_pooling2d_padding(x, ksize, stride) -> dynamic_padding
描述：  
- `x` `Tensor4D` 预计要进行 padding 的数据
- `ksize` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dynamic_padding`  `Int[4, 2]`输出的padding形式，为4x2维

参数：  
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `valid` `Int`  
非0数表示计算为：
```
output_height = floor((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
output_width = floor((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
```
0表示计算为：
```
output_height = ceil((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
output_width = ceil((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
```
- `padding` `Int[4, 2]` 静态进行padding的数据
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

### _onnx_pooling2d_padding(x, ksize, stride) -> dynamic_padding
描述：  
- `x` `Tensor4D` 预计要进行 padding 的数据
- `ksize` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dynamic_padding`  `Int[4, 2]`输出的padding形式，为4x2维

参数：  
- `auto_pad` `String` 为 `NOTSET`、`SAME_UPPER`、`SAME_LOWER`、`VALID`
`NOTSET`表示计算为：
```
output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
* pad_shape[i] is sum of pads along axis i
```
`VALID`表示计算为：
```
output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
```
`SAME_UPPER`和`SAME_LOWER`表示计算为：
```
output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
```
动态padding大小为：
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```
- `padding` `Int[4, 2]` 静态进行padding的数据
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

## 附录

1. 在做基础运算的时候，`x`和`a`有会三种意义，分别为`标量`，`张量`和`广播张量`。这里的广播张量的意义为：
   加入 x 的 shape 为 `[10, 10]` `a` 的 `shape` 为 `[10, 1]`，这个时候
   `y_{ij} = x_{ij} + a_{i0}` 在 `a` 的第二个维度，实现了广播。  
   注意： 广播的维度可以在矩阵中存在多份，默认维度大小为 `1` 的都支持广播。
   
2. `Pooling` 框架支持的 `padding` 类型。  
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
   caffe(mx_same):
   ```
   output_height = ceil((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
   output_width = ceil((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
   ```
   mx_valid(tf_valid):
   ```
   output_height = floor((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
   output_width = floor((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
   ```

