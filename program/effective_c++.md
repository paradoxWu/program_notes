# Effective c++
## 熟悉c++
1. c++是一个语言联邦，c++可以认为是c with classses，并且大部分时候大家都是这么使用的，但是c++不仅仅是多了类  
今天的c++是一个多重反范型编程语言(multiparadigm programming language),支持
>过程形式
>面向对象形式
>函数形式
>泛型形式
>元编程形式

为了更好的理解c++，我们需要从这四个方面了解
>1. c,c++是支持c的写法的
>2. object-oriented c++也就是c with classses
>3. Template c++,泛型编程
>4. STL库

## 尽量用编译器替换预处理器
具体的行为就是，prefer consts, enums, and inlines to #defines, 但是#define #include还是必需品

## 尽可能使用const
### const成员函数
```c++
class TextBlock{
public:
    const char& operator[](std::size_t index) const { return text[index]};
    char & operator[](std::size_t index) {return text[index]};
private:
    std::string text;
};
```

```c++
TextBlock tb("hello");
const TextBlock ctb("World");

std::cout<<tb[0]; //yes
tb[0] = 'x'; //yes
std::cout<<ctb[0];//yes
ctb[0] = 'x';//error
```
这样的作用就是创建一个const 类对象时候，能够给出对应的作用
> 成员函数名后加上`const` 关键字，能够在创建const对象时，调用对应的const成员函数（重载于非const的成员函数），但是需要注意的是，加上const关键字于函数名结尾的成员函数，内部是不可以修改这个类的non-static成员变量的(这是编译器bitwise const的约束)

如果想要释放掉上述的约束，那么可以把成员变量加上`mutable`的关键字
### 在const和non-const成员函数中避免重复
观察上述的类成员函数代码，其实我们可以看到两个重载`operator[]`的成员函数，他们的逻辑实现是一致的，只是return的类型有一个加上了const约束，为了减少代码重复，很显然我们可以做到一次实现两次调用，只要我们实现了常量性转移(casting away constness)
```c++
class TextBlock{
private:
    std::string text;
public:
    const char& operator[](std::size_t index) const { ... return text[index]};
    
    char & operator[](std::size_t index) {
       return
        const_cast<char&>(
            static_cast<const TextBlock&>(*this)[index]
        );
    };
};
```
我们在non-const的成员函数中加上了两个转型动作，一个用于调用同名const成员函数  
（如果没有这里的`static_cast<const TextBlock&>`，那么它将递归调用自己，进入一个死循环  
第二个`const_cast<char&>`则是用于去除常量性（return 类型去除const

>我们在non-const成员函数中调用了const成员函数用来实现减少重复代码。
但是反向做法--在const成员函数中调用non-const成员函数是不可以的，因为这不能保证逻辑上const（实际上你可能改变了其中的变量

## 确定对象使用前已经被初始化
c++中的变量没有初始化就被调用，总是会出现令人头痛的问题  
尽管STL中的容器之类保证了你没有初始化，它本身也会帮助你固定一个默认的初始化值，但是内置类型并不能保证这一点，我们需要遵循一个简单的原则
>确保每一个构造函数都将对象的每一个成员变量初始化
但是我们不能混淆赋值(assignment)与初始化(initialization)
```c++
Student::Student(const std::string &name,const int age)
{
    this->name_ = name;
    this->age = age;
}//这些都是赋值(assignment)而非初始化(initializations)
```
```c++
Student::Student(const std::string &name,const int age)
{
    this->name_ (name);
    this->age (age);
}//初始化
```
两个构造函数的结果是一致的，但是初始化的方式通常效率更高
(default constructor + operator = ) vs (copy constructor)
>此外，我们需要知道，c++有着十分固定的成员初始化次序，可以结合内存对齐的知识点来看

## 构造/析构/赋值
### c++对于类默默编写并调用了那些函数
default,copy构造函数，析构函数，copy assignment
>编译器产出的析构函数是非虚函数(non-virtual)，除非这个class的base class自身有virtual析构函数
### 如果不想使用编译器自动生成的函数，就要明确拒绝
我们可以将成员函数(构造析构，拷贝构造，拷贝复制)放在private下并且不予实现

### 为多态基类声明virtual析构函数
这是非常非常重要的一点，很多人都会知道构造函数不能为虚，析构可以，具体为什么呢？
在我们使用工厂模式的时候，我们会让base class的ptr去指向一个derived class的对象。如果base class的析构函数是non-virtual的，这是非常灾难的，因为它不能完全释放derived所构造出来的资源(一些派生类的成员变量)，形成一个诡异的"局部销毁"现象。**内存泄漏！！！**  
所以将virtual的析构函数就是允许各个derived class个性化自己的析构函数，并充分释放资源  
>但同时我们要注意，如果一个class并意图被当作一个base class时，我们最好不要令其析构函数为virtual，因为虚函数会占据额外的开销资源

**总结**：
1. 如果是一个base class或者带有任何virtual函数，那么它就应该拥有一个virtual析构函数
2. 如果class设计不是来作为base class的或者不是为了具有多态性，不应该声明virtual析构函数

### 令operator= 返回一个 reference to *this，以及考虑自我赋值
例如
```c++
class Age(){
    public:
        ... 
        Age & operator=(Age& rhs){
            ...
            return *this;
        }
    private:
        int age_;
}
```
> 这是一个好习惯，并非强制的，不遵循，编译一样没有问题，但是这样写是一个好习惯

上述代码中，我们可能出现这样的情况
```c++
Age a(10);
a = a;
```
尽管，这看上去很蠢，但是我们还是要避免它发生；三种方法：
1. 传统的方法
```c++
Age & operator=(Age rhs){
    if(this == &rhs) return *this;
    ...
    return *this;
}
```
2. 先删除，再构造
3. swap

## 资源管理
