##  张量数据类型
| python | PyTorch |
| :------: | :--: |
| int | intTensor of size() |
| float | FloatTensor of size() |
| int array | IntTensor of size[d1,d2,...] |
| float array | FloatTensor of size[d1,d2,...] |
| string | -- |

torch 中没有string类，但诸如nlp领域是需要处理这类数据的，如何解决？

> one-hot: [0,1,0,0]...此类编码方式
>
> embedding: World2vec glove

FloatTensor IntTensor ByteTensor  $\Rightarrow$ torch.cuda.xxxx

## 张量的创建
### tensor 与 numpy转换

`item()` 可以将一个tensor转换为一个python number
类似的还有如下

`numpy() from_numpy()` **该方法共享内存**
`torch.tensor()` **该方法为拷贝数据**

```python
# import from numpy
a=np.array([2,3.3])
print(torch.from_numpy(a))
b=np.ones([2,3])
print(torch.from_numpy(b))

# import from list
a=torch.tensor([2.,3.2])
b=torch.FloatTensor([2,3.2])
c=torch.tensor([[2.,3.2],[1.,22.3]])
print(a,b,c)

#full
print(torch.full([2,3],7.7)) #两行三列，全都为7.7
# arange 与np.arange(n)类似
print(torch.arange(0,10)) #左闭右开
print(torch.arange(0,10,2)) #第三个参数不畅
# linspace
print(torch.linspace(0,10,steps=4))#指定步数
# logspace x的指数次 指数为输入参数,可以指定底数为2,10,e
print(torch.logspace(0,1,steps=10))
# ones/zeros/eye
# randperm
print(torch.randperm(10)) #参数为(0,x)的上界x，开区间
# 可以用于随机抽取数据,index=torch.randperm(x)
```

## 索引

` y=x[0,:] `
**注意** 索引出来的结果并不是新生成的变量，仍然与原变量共享内存，故改变其中一者，另外一个也会变化

```python
# 图片格式数据索引示例
a=torch.rand(4,3,28,28)
print(a[0].shape)
print(a[0,0].shape)
print(a[0,0,2,4])

# 索引连续(切片)
# 两个切片 :x 从0连续选取x个
# x: 从x索引至最后
# -x: :-x 倒序
# 0:28:2 (起始-终止-步长) ::2 对全部范围以步长为2进行索引
print(a[:2].shape)
print(a[:2,:1,:,:].shape)

# 方法索引 index_select(x,torch.tensor([i,j]))
print(a.index_select(0,torch.tensor([0,2])).shape) #索引a下标0位置上[0,2)的样本
print(a.index_select(2,torch.arange(8)).shape) #索引a下标2位置上随机8个，即图片的行，从28行中随机选择8行

# ...符号 代表均选择
print(a[...].shape)
print(a[:,1,...].shape)

# select by mask
x=torch.randn(3,4)
mask=x.ge(0.5) #大于0.5的数字置为1，反之为0
print(torch.masked_select(x,mask),torch.masked_select(x,mask).shape)
```

## 维度变换
` view() `这个函数也是共享内存的，即产生的所谓新变量的变化仍然与原变量相关
` reshape() `该函数实现了真正新的副本，但是不能保证返回的是其拷贝(不推荐使用)
上述操作可以使用`clone`然后`view`

```python
#view reshape
a=torch.rand(4,1,28,28)
a.view(4,28*28) #view的原则是不改变变量的总大小numel() 适合全连接层
print(a.view(4,28*28).shape)
#但不要随意破坏数据存储的维度顺序，pytorch中的四维图片数据一般为[b,c,l,h]

# squeeze v.s. unsqueeze 挤压与展开
#unsqueeze在指定下标(正索引)前方插入一个维度，负数索引则为后方
print(a.unsqueeze(0).shape)
print(a.unsqueeze(-1).shape)
# squeeze() 不指定下标时，将为1的维度全部挤压，指定下标与unsqueeze操作一致

#expand /repeat
#expand：广播
#repeat: 内存复制/数据拷贝
b=torch.rand(1,32,1,1)
print(b.expand(4,32,14,14).shape)
print(b.expand(-1,-1,14,14).shape) #-1表示参数不变
print(b.repeat(4,1,14,14).shape) #每个参数对应拷贝的次数，结果为原数*参数

#tanspose 转置
#permute 用原下标指定新位置
```

1. torch.view(参数a，参数b，...)

参数a=3和参数b=2决定了将一维的tt1重构成3x2维的张量。

2. 有的时候会出现torch.view(-1)或者torch.view(参数a，-1)这种情况。

```text
>>> import torch
>>> tt2=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt2.view(-1)
>>> result
tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
```

由上面的案例可以看到，如果是torch.view(-1)，则原张量会变成一维的结构。

```text
>>> import torch
>>> tt3=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt3.view(2,-1)
>>> result
tensor([[-0.3623, -0.6115,  0.7283],
        [ 0.4699,  2.3261,  0.1599]])
```

由上面的案例可以看到，如果是torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度，在这个例子中a=2，tt3总共由6个元素，则b=6/2=3。

**1 torch.cat()**

> `torch.cat`(*tensors*,*dim=0*,*out=None*)→ Tensor

torch.cat()对tensors沿指定维度拼接，但返回的Tensor的维数不会变（默认dim=0）

```python3
>>> import torch
>>> a = torch.rand((2, 3))
>>> b = torch.rand((2, 3))
>>> c = torch.cat((a, b))
>>> a.size(), b.size(), c.size()
(torch.Size([2, 3]), torch.Size([2, 3]), torch.Size([4, 3]))
```

可以看到c和a、b一样都是二维的。

**2 torch.stack()**

> `torch.stack`(*tensors*,*dim=0*,*out=None*)→ Tensor

torch.stack()同样是对tensors沿指定维度拼接，但返回的Tensor会多一维

```python3
>>> import torch
>>> a = torch.rand((2, 3))
>>> b = torch.rand((2, 3))
>>> c = torch.stack((a, b))
>>> a.size(), b.size(), c.size()
(torch.Size([2, 3]), torch.Size([2, 3]), torch.Size([2, 2, 3]))
```

可以看到c是三维的，比a、b多了一维。

## Broadcast 

进行运算的前提条件是两个张量的shape size一致，如果不同，则

1. 会将小shape张量扩张与大shape一致。
2. 两个张量均扩张，例如[4,1]+[1,4] $\rightarrow$  [4,4]+[4,4]
### why broadcasting？
1. for actual demanding 
2. memory consumption
### Is it broadcasting-able?

**match from last dim**

> 1. if current dim=1,expand to the same
> 2. if either has no dim, insert one dim and expand to the same
> 3. otherwise, not broadcasting-able

### Merge or split

1. cat

```python
a=torch.rand(4,32,8)
b=torch.rand(5,32,8)
torch.cat([a,b],dim=0).shape
```
拼接两个张量，除了指定维度，其他维度shape需要一致
2. stack

```python
In:	a=torch.rand(4,4,32,8)
   	b=torch.rand(4,4,32,8)
   	torch.stack([a,b],dim=1).shape

out: torch.size([4,2,4,32,8])
```
   stack 会创建一个新的维度，用于区分原张量，原张量的shape必须完全一致

3. split

   按照指定每个拆分部分的维度size去拆分

  ```python
In: c=torch.rand(2,32,8) 
 	aa,bb=c.split([1,1],dim=0)
 	aa.shape, bb.shape
Out: torch.size([1,32,8]) torch.size([1,32,8])
  ```

4. chunk

   按照拆分出来的部分数量去拆分，下面例子中dim=0位置共2，拆成两部分，所以每部分都为1，如果用`c.chunk(2,dim=0)`就会报错，这是由于指定了每个部分都是2，但是原张量dim=0的size共2，不够拆分。
  ```python
In: c=torch.rand(2,32,8) 
 	aa,bb=c.chunk(2,dim=0)
 	aa.shape, bb.shape
Out: torch.size([1,32,8]) torch.size([1,32,8])
  ```
## 基本运算
add() sub() mul() div() (+ - * /)  方法与重载运算符等效
### 三种形式
` x+y  add(x,y)  y.add_(x) `

### 矩阵相乘三种形式

`Torch.mm Torch.matmul @` 推荐重载运算符 @ notice：torch.mm 仅仅适用于2d张量

```python
In: a=torch.rand(4,784)
	x=torch.rand(4,784)
	w=torch.rand(512,784) #torch默认以(channel.Out, channel.In)排序
	(x@w.t()).shape
Out: torch.Size([4,512])
```

这就是个实际可能的例子，线性层的降维 784 $\rightarrow$ 512, x@w<sup>t</sup> (4,784]*[784,512])

### 其他运算

1. `power() sqrt() rsqrt() 平方根 平方根导数` 与`**`等价

2. `torch.exp(), torch.log()` 

3. `.floor() .ceil()` 取小于它的最近整数， 大于它的最近整数(底与顶)

4. `.round()`四舍五入取整

5. `.trunc() .frac()` 拆分分别取整数与小数部分

6. gradient clipping  `.clamp()`

   设定最小值或范围`grad.clamp(min) grad.clamp(min,max)`

### 统计属性

norm mean sum prod max min argmin argmax

1. norm 范数 

   > norm-1:  $\sum xi$  
   >
   > norm-2: $\sqrt{\sum x_i^2}$ 
   >
   > `norm(p,dim=n)`指定在第n维上进行norm-p范数求解，指定哪个维度，哪个维度将被消掉

2. min max mean sum prod(累乘) 

3. argmax argmin(返回最大最小值的索引）

   可以指定维度 `arg(dim=n)` 不指定维度会先打平维度成一维数组范围一维数组中的索引
   
4. dim keepdim

   `keepdim=True` 可以保证维度不变，否则运算后指定维度会被消去

5. topk kthvalue

   > topk: a.topk(x,dim=n,largest=T/F)（返回前x大值，或前x小值(改largest=False))
   >
   > kthvalue(x,dim=n)(返回第x小的值)

6. torch.eq(a,b) torch.equal(a,b)

### 高阶操作

1. `torch.where(condition,x,y)` 满足条件输出来自x，不满足输出来自y

   <img src="D:\program files\WPS Office\WPS Cloud Files\cv\mine\pic\torch1.PNG" alt="torch1" style="zoom:50%;" />

2. `torch.gather(input,dim,index,out=None)` $\rightarrow$ Tensor  

   

### GPU使用

`if torch.cuda.is_available():  
    device = torch.device("cuda")  
    y = torch.ones_like(x,device=device)  
    x = x.to(device) # x.to("cuda")`  

## 梯度

### what is grad?

导数： derivate

偏微分: partial derivate

梯度：gradient

1. 变化速度
2. 变化方向

### How to search for minima?

new = now - learning rate * grad

**初始化**:有可能影响优化结果

**Learning rate**: 影响 收敛速度 与 收敛精度

**动量**: 帮助摆脱局部优解

### 自动求梯度

`torch.autograd.grad`

tensor包含一个属性`.grad`
链式法则中`.backward()`

链式法则即导数中求导(求梯度)的链式法则  

## 激活函数 activation functions

### sigmoid/ logistic

$$
f(x)= \sigma(x)=\frac{1}{1+e^{-x}}
$$

derivative: 
$$
\sigma^`=\sigma(1-\sigma)
$$

>
> 连续光滑，且值被压缩在0~1之间(rgb值常用)
>
> 缺点: 逼近无穷的地方，其导数逼近0，梯度得不到更新

```python
import torch
a=torch.linspace(-100,100,10)
torch.sigmoid(a)
#或者
from torch.nn import functional as F 
F.sigmoid(a)
```

### Tanh

$$
f(x)=tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}=2*sigmoid(2x)-1
$$

使用一致，`torch.tanh(a)`

### Rectified Linear Unit (ReLU)

$$
f(x)=\begin{cases}
0 & x<0\\
x & x\geq0
\end{cases}
$$

其导数非常的简单,减少了梯度爆炸与梯度离散情况

`torch.relu(a)`or`F.relu(a)`

### softmax

$$
S(y_i)=\frac{e^{y_i}}{\Sigma e^{y_j}}
$$

soft version of max (使大概率更大,小概率更小)

```python
a=torch.rand(3)
a.requires_grad_()
p=F.softmax(a,dim=0)
#p.backward()
torch.autograd.grad(p[1],[a],retain_graph=True)#保持图
torch.autograd.grad(p[2],[a])
```



## Loss 

### Mean Squared Error MSE 均方差

$$
loss=\Sigma[y-(xw+b)]^2 \\L2-norm = \lvert\lvert y-(xw+b)\rvert\rvert _2 \\loss = norm(y-(xw+b))^2
$$

`torch.norm(y-pred,2).pow(2)`

```python
from torch.nn import functional as F
import torch
x=torch.ones(1)
w=torch.full([1],2,requires_grad=True)
mse=F.mse_loss(torch.ones,x*w)
#求梯度
torch.autograd.grad(mse,[w])
#or
mse.backward()
w.grad
```



### Cross Entropy loss

### 注意两者的区别

MSELoss（）多用于回归问题，也可以用于one_hotted编码形式，

CrossEntropyLoss()名字为交叉熵损失函数，不用于one_hotted编码形式

MSELoss（）要求batch_x与batch_y的tensor都是FloatTensor类型

CrossEntropyLoss（）要求batch_x为Float，batch_y为LongTensor类型

### Optimizer

`import torch.optim as optim`

#### [Adam](https://blog.csdn.net/weixin_39228381/article/details/108548413)

```python
torch.optim.Adam(params,
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)
```

> params : parameters of the model such as `model.parameters()`
>
> lr: learning rate
>
> betas 平滑常数
>
> eps 加在分母上防止除0
>
> weight_decay的作用是用当前**可学习参数p**的值修改偏导数 **weight_decay的作用是L2正则化，和Adam并无直接关系。**
>
> 如果amsgrad为True：保留历史最大的 **累计梯度的平方 **  **amsgrad和Adam并无直接关系。**

#### SGD





## TensorBoard 可视化

`pip install tensorboardX`

cmd中首先运行tensorboard

```cmd
tensorboard --logdir [dir]
```

[dir]为路径

```python
import tensorboard

writer = SummaryWriter(log dir ="./logs",flush_secs=xxx) 

## 图
if Cuda:
    ...xxx.cuda()
else:
    graphs = torch.from_numpy(...)
    
writer.add_graph(model,(graphs,))

## loss
writer.add_scalar('title_name',object,(batch_size*epoch+iteration))
```

> 第一个参数可以简单理解为保存图的名称
>
> 第二个参数是可以理解为Y轴数据，
>
> 第三个参数可以理解为X轴数据。
>
> 当Y轴数据不止一个时，可以使用writer.add_scalars().

http://localhost:6006/

# Record

## ***Day 1***

### 环境安装

1.  以下均在 **wsl2 ubuntu 18.04** 中实现
    首先安装Miniconda用于管理环境 [Minicoda](（https://blog.csdn.net/weixin_44159487/article/details/105620256)

```
conda create -n studyTorch python=3.6.9
activate studyTorch
```

可以选择翻墙或者修改[镜像源](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)

2. 安装以下 pkg

```
conda install torch #无gpu版本  
conda isntall numpy
conda install matplotib
```

今天被 *wsl* 的网络搞崩溃了 感觉还是纯 *linux* 或者 *windows* 好用  最终使用pip镜像安装解决

**wsl中直接调用jupyter会出错，这是由于wsl没有浏览器造成的，使用如下调用方式**
`jupyter notebook --no-browser`

## ***Day 2***

### 概念学习

#### 数据表示

> 张量(tensor): 矩阵向任意维度(dimension)的推广,通常叫做轴(axis)，阶(rank)，不过有时称呼为维度(dimension)
> 标量(0D张量)(scalar)  
> 向量(1D张量)(vector）
> 矩阵(2D张量)(matrix)
> 3D以及以上

#### 关键属性

> 轴的个数(阶)：ndim
>  形状：shape 沿每个轴的个数(tip: 标量.shape = none)
>  数据类型：dtype (float32,float64,uint8,极少数情况下可能为char)

### 现实中的数据轴数量

>通常第一个轴(0轴)是样本轴，有时也叫样本维度，样本通常非常多，此时就需要分批(batch)，这是一个经常遇到的概念，batch axis batch dimension batch_size


向量数据(2D): (samples,features)
序列数据(3D): (samples,timesteps,features)
图像(4D): (samples,height,width,channels) or (samples,channels,height,width)
视频(5D): (samples,frames,height,width,channels) or (samples,frames,channels,height,width)

>2D：例如居民数据，有一千个居民，数据包括每个人的年龄，性别，收入，那么featurs=3,examples=1000,so the shape of this training data is (1000,3)

>3D:股票股价数据，包括当前价格，前一分钟的最高价，前一分钟的最低价，features=3,假设每天交易时间共390分钟,则整个交易日被保存在(390,3)的向量中，而200天的交易数据就形成了shape为(200,390,3)

>4D：图像数据，通常图像数据具有大小(height * width)和颜色通道(channels),例如128张 256 * 256的灰度图就保存的一个shape=(128,256,256,1)的张量中，有以下两个原则
>
>>1.channels-last(tensorflow):通道在最后
>>2.channels-first(theano,torch)：通道在examples后

>5D:视频数据可以视为连续的图片数据，每一帧可以用一个3D张量保存，而视频就是加上帧数之后的4D张量，再加上样本就成了了5D张量
>(samples,frames,height,width,channels)

## ***Day 3***

### 张量运算(Tensor Operation)

#### 逐元素运算

> 独立地对张量的每一个元素运算

#### 广播

> shape 不同的张量运算时，shape小的张量会被**广播**来匹配大shape张量运算
> 1 添加轴，使轴数目相等 ndim
> 2 较小的张量沿新轴的方向重复

#### 点积

>部分库中 * 实现**逐元素**乘积，numpy与keras使用dot()函数实现**点积**

```
import numpy as np
z = np.(x,y)
```

两个向量之间的点积是一个标量，而且只有元素个数相同的向量之间才能做点积。注意，如果两个张量中有一个的 ndim 大于 1，那么 dot 运算就不再是对称的，也就是说，dot(x, y) 不等于 dot(y, x)

#### 张量变形(reshape)

>图像输入预处理时经常使用,reshape改变变量的行与列 --> 得到想要的形状，但是元素的总个数不变
>一种特殊的变形 -- 转置 transpose()

### data type

torch.floatTensor (int,byte,double)

#### 几个.方法

> dim() --> rank
> shape() --> 形状
> Size()  --> 总的大小