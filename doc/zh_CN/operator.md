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

