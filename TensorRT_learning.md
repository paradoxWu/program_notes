# TensorRT

## 环境要求

1. conda(annaconda/miniconda)(推荐python3.9)
2. pytorch(推荐1.9)
3. cuda11.4
4. cudnn8.2.4
5. TensorRT8.2.1.8
6. protobuf3.11.4
7. opencv-4.2.0
   
安装教程见[https://zhuanlan.zhihu.com/p/446477459]()

## 导出onnx并用c++推理
对于pytorch模型，torch本身提供了接口`torch.onnx.export`

例如
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=True)
        self.conv.weight.data.fill_(0.3)
        self.conv.bias.data.fill_(0.2)

    def forward(self, x):
        x = self.conv(x)
        # return x.view(int(x.size(0)), -1)
        return x.view(-1, int(x.numel() // x.size(0)))

model = Model().eval()

x = torch.full((1, 1, 3, 3), 1.0)
y = model(x)
print(y)

torch.onnx.export(
    model, (x, ), "lesson1.onnx", verbose=True
)
```

### onnx导出要点
1. 对于shape或size需要加上int强制类型转换例如不应该使用`tensor.view(tensor.size(0),-1)`，而是`tensor.view(int(tensor.size(0)),-1)`
2. 对于nn.Upsample或nn.functional.interpolate函数，使用scale_factor指定倍率而不是size参数
3. 一般情况，我们只能讲batch维度设置成-1，例如reshape，view，将batch维度设置成-1，其他维度进行计算
4. torch.onnx.export指定dynamic_axes参数且只指定batch维度，不指定其他维度，对于动态宽高，选择其他方案解决

>这些做法主要是为了简化过程复杂度，去掉gather，shape类的节点

### trtc++调用
```cpp
static void lesson1(){

    /** 模型编译，onnx到trtmodel **/
    TRT::compile(
        TRT::Mode::FP32,            /** 模式, fp32 fp16 int8  **/
        1,                          /** 最大batch size        **/
        "lesson1.onnx",             /** onnx文件，输入         **/
        "lesson1.fp32.trtmodel"     /** trt模型文件，输出      **/
    );

    /** 加载编译好的引擎 **/
    auto infer = TRT::load_infer("lesson1.fp32.trtmodel");

    /** 设置输入的值 **/
    infer->input(0)->set_to(1.0f);

    /** 引擎进行推理 **/
    infer->forward();

    /** 取出引擎的输出**/
    auto out = infer->output(0);
}
```
### 动态batch，动态宽高的处理方式
1. 动态batch
静态batch按照固定batch推理，耗时固定
> 导出onnx模型需要batch不固定，通常写-1；通常可以指定dynamic_axes

2. 动态宽高
onnx模型导出指定了宽高，trt的推理引擎大小也固定
需要动态的宽高，可以再trt中实现，也可以在onnx中实现；但trt中实现比较复杂(trt中实现其实是需要不同的onnx文件的，即需要从多个pytorch模型导出onnx)

- TRT::compile函数的inputsDimsSetup参数重定义输入的shape
- TRT::set_layer_hook_reshape钩子动态修改reshape参数实现适配

```cpp
    /** hook函数**/
    TRT::set_layer_hook_reshape(
        [&](const string& name,const vector<int64_t>& shape)->vector<int64_t>{
            INFO("name: %s, shape: %s",name.data(),iLogger::join_dims(shape).data());
            return {-1,25};
        }
    );//当有不同需求时，可以根据name来return不同结果

    /** 模型编译，onnx到trtmodel **/
    int32_t height = 5, width = 5;
    TRT::compile(
        TRT::Mode::FP32,            /** 模式, fp32 fp16 int8  **/
        1,                          /** 最大batch size        **/
        "lesson1.onnx",             /** onnx文件，输入         **/
        "lesson1.fp32.trtmodel",    /** trt模型文件，输出      **/
        {{1,1,height,width}}        /** 对输入shape重定义    **/
    );
```
## 实现一个自定义插件
1. 对需要插件的layer，写一个类A，继承自torch.autograd.Function
2. 对这个类增加静态方法@staticmethod `symbolic`,其中返回`g.op`
   ```python
   class HSwishImplementation(torch.autograd.Function):

    # 主要是这里，对于autograd.Function这种自定义实现的op，只需要添加静态方法symbolic即可，除了g以外的参数应与forward函数的除ctx以外完全一样
    # 这里演示了input->作为tensor输入，bias->作为参数输入，两者将会在tensorRT里面具有不同的处理方式
    # 对于附加属性（attributes），以 "名称_类型简写" 方式定义，类型简写，请参考：torch/onnx/symbolic_helper.py中_parse_arg函数的实现【from torch.onnx.symbolic_helper import _parse_arg】
    # 属性的定义会在对应节点生成attributes，并传给tensorRT的onnx解析器做处理
    @staticmethod
    def symbolic(g, input, bias):
        # 如果配合当前tensorRT框架，则必须名称为Plugin，参考：tensorRT/src/tensorRT/onnx_parser/builtin_op_importers.cpp的160行定义
        # 若你想自己命名，可以考虑做类似修改即可
        #
        # name_s表示，name是string类型的，对应于C++插件的名称，参考：tensorRT/src/tensorRT/onnxplugin/plugins/HSwish.cu的82行定义的名称
        # info_s表示，info是string类型的，通常我们可以利用json.dumps，传一个复杂的字符串结构，然后在CPP中json解码即可。参考：
        #             sxai/tensorRT/src/tensorRT/onnxplugin/plugins/HSwish.cu的39行
        return g.op("Plugin", input, bias, name_s="HSwish", info_s=json.dumps({"alpha": 3.5, "beta": 2.88}))
   ```
3. 对类A增加`forward`静态方法
4. 实现一个OP的类，继承自`nn.Module`，在OP.forward中调用A.apply
   ```python
   class MemoryEfficientHSwish(nn.Module):
    def __init__(self):
        super(MemoryEfficientHSwish, self).__init__()
        
        # 这里我们假设有bias作为权重参数
        self.bias = nn.Parameter(torch.zeros((3, 3, 3, 3)))
        self.bias.data.fill_(3.15)

    def forward(self, x):
        # 我们假设丢一个bias进去
        return HSwishImplementation.apply(x, self.bias)
   ```
5. 正常使用OP集成到模型中即可
   ```python
    class FooModel(torch.nn.Module):
        def __init__(self):
            super(FooModel, self).__init__()
            self.hswish = MemoryEfficientHSwish()

        def forward(self, input1, input2):
            return F.relu(input2 * self.hswish(input1))
   ```
### 编译/推理环节
1. src/tensorRT/onnxplugin/plugins中写cu和hpp文件
2. 自定义config
3. 实现类继承自TRTPlugin，实现config用于返回自定义类，getOuputDimensions返回layer处理后的tensor大小，enqueue实现具体推理工作
