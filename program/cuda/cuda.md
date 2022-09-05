硬件知识部分略讲
## CUDA结构基本组成
这里我们只需要知道`grid`,`block`,`thread`即可  
首先是对应关系
线程处理器 (SP) 对应线程 (thread)。  
多核处理器 (SM) 对应线程块 (thread block)。  
设备端 (device) 对应线程块组合体 (grid)。  
thread组成block，block组成grid  
1. thread
> GPU一个核处理一个thread，位于同一个block中的thread是可以同步、通信的
2. block
> 线程组成了block，block内部有shared memory（共享内存）是local的（局部)，在单个block内部可同步通信，而各个block之间是独立的，并行的，整体是global的使用的是显存
3. grid
>grid 由block组成 
4. 线程束 (warp)
> 一个线程束是32个线程的集合，所以往往显存是32的倍数.  
SM 采用的 SIMT (Single-Instruction, Multiple-Thread，单指令多线程) 架构，warp (线程束) 是最基本的执行单元。一个 warp 包含32个并行 thread，这些 thread 以不同数据资源执行相同的指令。一个 warp 只包含一条指令，所以：warp 本质上是线程在 GPU 上运行的最小单元。

每个 thread 都有自己的一份 register 和 local memory 的空间。同一个 block 中的每个 thread 则有共享的一份 share memory。此外，所有的 thread (包括不同 block 的 thread) 都共享一份 global memory。不同的 grid 则有各自的 global memory。 

## 线程块 id & 线程 id：定位独立线程的门牌号

核函数需要确定每个线程在显存中的位置，我们之前提到 CUDA 的核函数是要在设备端来进行计算和处理的，在执行核函数时需要访问到每个线程的 registers (寄存器) 和 local memory (局部内存)。在这个过程中需要确定每一个线程在显存上的位置。所以我们需要使用线程块的 index 和线程的 index 来确定线程在显存上的位置。

## CUDA程序的基本步骤
一个 CUDA 程序，我们可以把它分成3个部分：

第1部分是：从主机 (host) 端申请 device memory，把要拷贝的内容从 host memory 拷贝到申请的 device memory 里面。  

第2部分是：设备端的核函数对拷贝进来的东西进行计算，来得到和实现运算的结果，图4中的 Kernel 就是指在 GPU 上运行的函数。  

第3部分是：把结果从 device memory 拷贝到申请的 host memory 里面，并且释放设备端的显存和内存。  

## kernel 函数。

核函数调用的注意事项

1. 在 GPU 上执行的函数。
2. 一般通过标识符 __global__ 修饰。  
3. 调用通过<<<参数1,参数2>>>，用于说明内核函数中的线程数量，以及线程是如何组织的。  
4. 以网格 (Grid) 的形式组织，每个线程格由若干个线程块 (block) 组成，而每个线程块又由若干个线程 (thread) 组成。  
5. 调用时必须声明内核函数的执行参数。  
6. 在编程时，必须先为 kernel 函数中用到的数组或变量分配好足够的空间，再调用 kernel 函数，否则在 GPU 计算时会发生错误。  

>调用函数的时候会在后面写上<<< >>>，意思这是从host端到device端的内核函数调用，里面的参数是执行配置，用来说明使用多少线程来执行内核函数
GridDim = Block_num
BlockDim = Thread_num


## CUDA 编程的标识符号

不同的表示符号对应着不同的工作地点和被调用地点。核函数使用 __global__ 标识，必须返回 void。__device__ & __host__ 可以一起用。

_global_
>被__global__函数类型限定符修饰的函数被称为内核函数，**该函数在host上被调用，在device上执行**，只能返回void类型，不能作为类的成员函数。调用__global__修饰的函数是异步的，也就是说它未执行完就会返回。

_device_
>被__device__函数类型限定符修饰的函数只能在device上被调用，在device上执行，用于在device代码中内部调用。

_host_
>被__host__函数类型限定符修饰的函数只能在host上被调用，在host上执行，也就是host上的函数，__host__函数类型限定符可以省略。