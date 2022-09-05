### `np.expand_dims()`

```python
np.expand_dims（a，axis=x）
```

在a的第x轴维度上扩展一个维度

例如

```python
a = np.array([[1,2],[3,5]])
y = np.expand_dims(a, axis=2)
z = np.expand_dims(a, axis=1)
print(a.shape)
print(y.shape)
print(z.shape)
```
输出
(2, 2)
(2, 2, 1)
(2, 1, 2)


### `np.tile(x,(a,b))`

将x沿着y轴扩展为a倍，x轴扩展为b倍，为1时不变。

###  `np.pad()`
[np.pad()](https://blog.csdn.net/zenghaitao0128/article/details/78713663)

1. 语法结构
> pad(array, pad_width, mode, **kwargs)
> 返回值：数组

2. 参数解释
> array : 填充的数组
> pad_width: 填充的数目按轴的顺序填充(见下例)
> mode: 填充方式
3. 填充方式
>constant’——表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0
‘edge’——表示用边缘值填充
‘linear_ramp’——表示用边缘递减的方式填充
‘maximum’——表示最大值填充
‘mean’——表示均值填充
‘median’——表示中位数填充
‘minimum’——表示最小值填充
‘reflect’——表示对称填充
‘symmetric’——表示对称填充
‘wrap’——表示用原数组后面的值填充前面，前面的值填充后面

例子:

常数填充法(在卷积神经网络中，通常采用constant填充方式!!)

下面的例子中 array =A, pad_width =((3,2)(2,3)),mode= 'constant'

即对A 数组进行填充，填充范围为第0轴(此例子中的行)，原数组在行这个维度上填充前3行，后2行，这就是第一个(3,2)的意义，同理，在第1轴上填充前方2列，后方3列，采用常规填充，填充值指定为0

```python
import numpy as np
A = np.arange(95,99).reshape(2,2) 
np.pad(A,((3,2),(2,3)),'constant',constant_values = (0,0)) 
#在数组A的边缘填充constant_values指定的数值
#（3,2）表示在A的第[0]轴填充（二维数组中，0轴表示行），即在0轴前面填充3个宽度的0，比如数组A中的95,96两个元素前面各填充了3个0；在后面填充2个0，比如数组A中的97,98两个元素后面各填充了2个0
#（2,3）表示在A的第[1]轴填充（二维数组中，1轴表示列），即在1轴前面填充2个宽度的0，后面填充3个宽度的0
#constant_values表示填充值，且(before，after)的填充值等于（0,0）
```

```bash
> array([[95, 96],
       [97, 98]])
       
> array([[ 0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0],
       [ 0,  0, 95, 96,  0,  0,  0],
       [ 0,  0, 97, 98,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0]])    
```

