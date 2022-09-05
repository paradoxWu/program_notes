# Java

编译型
>源程序编译后即可在该平台运行 运行速度快 c/c++

解释型
>在运行期间才编译 跨平台性好 python

Java和其他的语言不太一样。因为java针对不同的平台有不同的JVM，实现了跨平台。所以Java语言有一次编译到处运行的说法。

1.**你可以说它是编译型的：**因为所有的Java代码都是要编译的，.java不经过编译就什么用都没有。 
2.**你可以说它是解释型的：**因为java代码编译后不能直接运行，它是解释运行在JVM上的，所以它是解释运行的，那也就算是解释的了。 
3.但是，现在的JVM为了效率，都有一些JIT优化。它又会把.class的二进制代码编译为本地的代码直接运行，**所以，又是编译的。**

## 版本

jdk: java development kit

jre: java runtime environment

jvm: java virtual machine

jdk8 + os写入

### IDEA
vscode写java很折磨,下载IDEA
### vscode

## 语法

### 注释
### 标识符
### 数据类型
强类型语言: 变量必须先定义才能使用
弱类型语言
**Java两大类**
1. 基本类型：数值类型 boolean
2. 引用类型：类，接口，数组
### 类型转换
强制类型转换 （高$ \rightarrow$ 低）`(type) variable_name` e. g. :`(char)a` 
自动类型转换   （低$\rightarrow$ 高）
tips：

1. 不能对boolean转换
2. 高精度至低精度 强制
3. 转换可能发生内存溢出，或者精度问题

### 变量

1. 局部变量：在一个函数内部
2. 类变量: static 前缀，不需要通过调用实例对象，可以直接引用类变量
3. 实例变量: class内属性，继承对象属性，如果不初始化，基本类型默认为0，false，其他类型为null
4. constant：常量 `final variable_name` 通常常量大写字母命名

5. 命名规范:

   > 类成员：首字母小写+驼峰原则 **lastName**
   >
   > 局部变量： 首字母小写+驼峰原则 **monthSalary**
   >
   > 常量：全部大写+下划线 **MAX_VALUE**
   >
   > 类名：首字母大写+驼峰原则 **Man GoodMan**
   >
   > 方法名：首字母小写+驼峰原则 **run(),runRun()**

### 运算符

一般运算符与c一致

字符串连接符 + string

```java
System.out.println("string"+a+b)
```

### 包机制

```java
package pkg1.pkg2...
import package
```

### Scanner 对象

java.util.Scanner

```java
import java.util.Scanner;

public class Base {
    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        System.out.println("使用next方式接收：");

        if(scanner.hasNext()){
            String str = scanner.next();
            System.out.println("输入内容为："+ str);
        }
        scanner.close();
    }
}
```

通过`scanner.next()`or`scanner.nextLine()`方法获取输入内容，可以使用`hasNext() hasNextLine()`方法判断是否有输入。 可以判断输入类型`hasNextInt(), hasNextFloat(),nextInt(),nextFloat()` 

>next(): 
>1. 读到有效字符后才可以结束输入
>2. 有效字符前的空白会被自动忽略
>3. next() 不能得到带有空格的字符串
>
>nextLine():
>1. 以Enter为结束符
>2. 可以获得空白，空格

### 方法

类似于函数

### 可变参数

```java
int ... i
```

声明类型后 可以通过在变量名前加...形式变为可变参数

### 数组

定义方式1.`dataType[] variable_name;`

定义方式2. `dataType variable_name[];`// c++ style

创建方式: `variable_name = new dataType [size]`

```java
int [] grades = new int [50]
int [][]array = new int [2][5]//多维数组
```
数组自带属性.length

### Arrays

Arrays.toString(var_name)

Arrays.sort(var_name)

## Java 内存分析

**堆**：1.存放new的对象和数组 (包含具体值)2. 可以被所有线程共享，不会存放在别的对象引用

**栈**：1.存放基本变量类型 2.引用对象的变量

**方法区**：包含了所有的class和static变量

## 面向对象(Object-Oriented Programming)

封装，继承，多态

**类** $\rightarrow$ 实例化**对象** 