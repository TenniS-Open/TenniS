# 算符支持

----

这里列举出，框架已经支持的算符，以及对应的参数设置。

所有内置算符都包含参数：
- `#op`: `string` 算符的名称
- `#name`: `string` 实例化名称
- `#output_count`: `int` 输出结果数


## 内置算符

### _field
描述：用于获取 `Packed Tensor` 的元素。  
输入：1-`Packed Tensor`  
输出：1-tensor  
参数：
- `offset`: `uint` `[Required]` 要取输入的 `Packed Tensor` 的偏移下标

### _pack
描述：打包输入元素，输出 `Packed Tensor` 
输入：N-tensor  
输出：1-`Packed Tensor`  
参数：  

### _
