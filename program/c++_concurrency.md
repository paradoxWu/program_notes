## 线程管理

c++11后对于线程的管理都内置在了标准库中
我们需要 `#include <thread>`这个头文件
管理 `std::thread`这个对象

### 启动线程

通过实例化一个 `std::thread`的instance，我们就启动了线程

```c++11
#include <thread>
void f()
{
    std::cout<<"concurrency thread1"<<std::endl;
}

int main()
{
    std::thread thread1(f); //这个线程在这里就开始了
    return 0;
}
```

这种函数在其所属线程上运行，直到函数执行完毕，线程也就结束了。在一些极端情况下，线程运行时，任务中的函数对象需要通过某种通 讯机制进行参数的传递，或者执行一系列独立操作;可以通过通讯机制传递信号，让线程停 止。线程要做什么，以及什么时候启动，其实都无关紧要。总之，使用C++线程库启动线程， 可以归结为构造 std::thread 对象
同时，std::thread 可以用可调用类型构造，将带有函数调用符类型的实例传入 std::thread 类中，替换默认的构造函数

```c++11
class background_task 
{
    public: void operator()() const 
    { 
        do_something(); 
        do_something_else();
    } 
};
background_task f; 
std::thread my_thread(f);
```

**一个重点细节：** 当把函数对象传入到线程构造函数中时，需要的是一个命名的变量而不是一个临时对象

```c++11
background_task f; 
std::thread my_thread(f); //启动线程
std::thread my_thread(background_task());//返回一个 std::thread 对象的函数，而非启动了一个线程
```

使用在前面命名函数对象的方式，或使用多组括号1，或使用新统一的初始化语法2，可以避免这个问题

```c++11
std::thread my_thread((background_task())); // 1 
std::thread my_thread{background_task()}; // 2
```

使用lambda表达式也能避免这个问题。lambda表达式是C++11的一个新特性，它允许使用一 个可以捕获局部变量的局部函数(可以避免传递参数，参见2.2节)。想要具体的了解lambda表 达式，可以阅读附录A的A.5节。之前的例子可以改写为lambda表达式的类型：

```c++11
std::thread my_thread([]{ 
        do_something(); 
        do_something_else(); 
    });
```

#### 传入参数

某些情况下，启动的并发函数是需要传入一些外部参数的，这里我们直接在thread初始化时，传入即可

```c++11
std::thread my_thread(f,arg1,arg2,...);
```

需要注意的是，thread的构造函数并不会考虑线程函数期望的参数类型（值传递，引用传递），而是盲目的进行拷贝 值传递，如果我们需要进行引用传递的话，需要使用std::ref(),如下

```c++11
std::thread my_thread(f,std::ref(arg1));
```

### 线程结束

在main()函数中启动一个并发线程，如果不等待这个线程结束，而直接继续并发执行，就会导致因为主函数的线程结束而导致子线程提前结束，所以需要一个阻塞等待
往往使用 `thread.join()`方法，另外 `thread.detach()`方法可以将子线程从主线程独立开，主线程结束后，子线程继续，但是这个是无法解决main()提前结束的问题的，同时，工作中通常使用join而不是detach

#### 等待线程完成

join()是简单粗暴的等待线程完成或不等待。当你需要对等待中的线程有更灵活的控制时，比 如，看一下某个线程是否结束，或者只等待一段时间(超过时间就判定为超时)。想要做到这 些，你需要使用其他机制来完成，比如条件变量和期待(futures)
调用join()的行为，还清理了线程相关的存储部分，这样 std::thread 对象将不再与已经 完成的线程有任何关联。这意味着，只能对一个线程使用一次join();一旦已经使用过 join()， std::thread 对象就不能再次加入了，当对其使用joinable()时，将返回false。

#### 后台运行线程

使用detach()会让线程在后台运行，这就意味着主线程不能与之产生直接交互。也就是说，不 会等待这个线程结束；如果线程分离，那么就不可能有 std::thread 对象能引用它，分离线程 的确在后台运行，所以分离线程不能被加入。不过C++运行库保证，当线程退出时，相关资源 的能够正确回收，后台线程的归属和控制C++运行库都会处理

### 转移线程所有权

C++标准库中有很多资源占有(resource-owning) 类型，比如 std::ifstream ， std::unique_ptr 还有 std::thread 都是可移动，但不可拷贝。 这就说明执行线程的所有权可以在 std::thread 实例中移动，下面将展示一个例子。例子中， 创建了两个执行线程，并且在 std::thread 实例之间(t1,t2和t3)转移所有权

```c++11
void some_function(); 
void some_other_function(); 
std::thread t1(some_function); // 1 
std::thread t2=std::move(t1); // 2 
t1=std::thread(some_other_function); // 3 
std::thread t3; // 4 
t3=std::move(t2); // 5 
t1=std::move(t3); // 6 赋值操作将使程序崩溃
```

### 运行时决定线程数量

std::thread::hardware_concurrency() 在新版C++标准库中是一个很有用的函数。这个函数会 返回能并发在一个程序中的线程数量。例如，多核系统中，返回值可以是CPU核芯的数量。 返回值也仅仅是一个提示，当系统信息无法获取时，函数也会返回0。但是，这也无法掩盖这 个函数对启动线程数量的帮助

## 线程间共享数据

- 共享数据带来的问题
- 使用互斥量保护数据
- 数据保护的替代方案

### 条件竞争

不同线程抢着完成同一个任务，而且往往涉及到公共数据的写
如果只读的话，是良性的，没有的问题

**避免的方法**：

1. 保证每个不变量保持稳定 （无锁编程）
2. 事务的方式去更新数据
   基本方式 c++标准库提供的互斥量！

### 互斥量保护共享数据
将所有访问共享数据结构的代码都标记为互斥，这样，任何一个线程在执行时，其他线程试图访问共享数据时，就必须进行等待。除非该线程就在修改共享数据，否则任何线程都不可能会看到被破坏的不变量。
当访问共享数据前，将数据锁住，在访问结束后，再将数据解锁。线程库需要保证，当一个 线程使用特定互斥量锁住共享数据时，其他的线程想要访问锁住的数据，都必须等到之前那 个线程对数据进行解锁后，才能进行访问。这就保证了所有线程都能看到共享数据，并而不破坏不变量  
需要编排代码来保护数据的正确性，并避免接口间的竞争条件也非常重要。不过，互斥量自身也有问题，也 会造成死锁，或对数据保护的太多(或太少)。

```c++11
# include <iostream>
# include <thread>
# include <mutex>

std::mutex mtx;

int globalVariable = 0;

void task1(){
    for(int i =0;i<100000;i++){
        mtx.lock();
        globalVariable++;
        globalVariable--;
        mtx.unlock();
    }
}

int main()
{
    std::thread t1(task1);
    std::thread t2(task1);
    std::cout<<globalVariable<<std::endl;
    t1.join();
    t2.join();
    return 0;
}
```
并不推荐这种写法，因为lock上锁后如果该线程出现了异常，而没有走到unlock处，就会出现死锁，别的地方也无法访问共享数据了  
推荐
```c++11
void task1(){
    for(int i =0;i<100000;i++){
        std::lock_guard<std::mutex> lock(mtx);
        globalVariable++;
        globalVariable--;
    }
}
```
除了`std::lock_guard`以外还有`std::unique_lock`可以使用，它拥有更自由的成员函数，用于提前解锁
### 死锁
RAII (Resource Acquisiton Is Initialization)

上锁顺序不同