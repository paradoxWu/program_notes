# 商汤

## C++相关

## GIT相关

### 拉取推送

ssh 方式需要注意密钥是否添加

```shell
git clone ssh/url
git pull 不推荐使用 
建议使用：git fetch -p + git merge
git push
git remote prune origin

```

### 分支管理

```shell
#在原有分支上开发新功能或debug
git checkout -b debug/dev_branch_name
```

```shell
#git rebase 将别的分支合并进来
git checkout original_branch
git rebase target_branch
#if冲突 then解决冲突 git rebase --continue
#else 不处理
git checkout target_branch
git merge original_branch
```

```
git rebase --onto [commit_id1] [commit_id2](不包含左，包含右)
```

```shell
#重命名远端/本地分支
git branch -m new_branch_name #本地重命名
git push origin --delete old_branch_name #删除远端repo分支
git push origin new_branch_name
git branch --set-upstream-to origin/new_branch_name
```

```shell
#修改commit信息
git log #查看commit记录
git rebase -i (commit id1, commit id2]#修改id1到id2之间的commit记录，不包括1，包括2
```

## Docker相关

`docker image ls [options]`查看所有已经拉取的镜像

> -q: 仅列出commit ID
> --filter：
> --format:

`docker image rm [image_name/ID]`删除镜像

### 创建/进入

docker run \[OPTIONS\] image \[COMMADN\] \[ARGs\]
OPTIONS具体说明
-it: -i 交互式操作 -t 终端
--rm 容器推出后随即将之删除 避免浪费空间
-e/--env list 设置环境变量
--net
--ipc --ipc=host：容器与主机共享内存，或容器间共享内存
--gpus 向容器内增加的显卡数量，可以指定 可以使用'all'
-v/--volume list 绑定目录, 这样docker外操作该文件夹，docker内也可以看到
-w/--workdir 指定容器内工作目录的名字

```shell
export BUILDER=registry.sensetime.com/kestrel_tatraffic/kestrel_tatraffic:kestrel_all_in_one_1.2.21

docker run --gpus '"device=0"' -it --rm --net=host --ipc=host \
                    -e DISPLAY=$DISPLAY \
                    --privileged \
                    -e XAUTHORITY=$XAUTH \
                    -u ${user_option} \
                    --entrypoint=bash \
                    --ulimit core=-1 \
                    --security-opt seccomp=unconfined \
                    -v ${PWD}:/flock_pipline \
                    -w /flock_pipline \
                    ${BUILDER}
```

### 文件传递

docker cp original_dir target_dir
容器可以使用 `ID:dir`的形式

### Dockerfile

可以通过Dockerfile定制镜像

```
FROM [image_name]
RUN <command>
```

> 要注意的是Docker的RUN命令每一层是互相叠加的，每Run完一次之后会提交一个commit ID，导致镜像臃肿， 正确的写法是 多使用 `&&` 连接命令

### Image相关

通过 `docker image ls`查看已经存在的所有image基，这与 `docker ps`不同，`docker ps`查看的是在运行中的容器，容器是可以在一个已commit的镜像基上创建的。
`docker image rm`可以删除指定镜像，但如果有其镜像创建的容器在运行中，会提示无法删除，谨慎使用 `-f`参数强制删除

```shell
# image commit
docker commit -m="has update" -a="runoob" [cotainer ID] repo:tag
```

-m：添加描述
-a：指定作者
repo:tag: 指定镜像仓库：镜像名

## Unit Test单元测试相关

## 性能测试相关

## 内存检查相关

```
valgrind --tool=memcheck --log-file=[log_name] --leak-check=yes --track-origins=yes + 可执行文件（参数）
```

```
cuda-memcheck --report-api-errors no --show-backtrace yes --leak-check full +可执行文件（参数） + （重定向）2>&1 | tee log_name
```

## CI相关

## Linux相关

直接在目录下查询包含[keywords]的文件以及行数 `grep -nr "查询的内容" [指定的目录路径或文件]`
e.g. `grep -nr "local_box" *` 即在当前目录下查询所有文件是否包含“local_box"这个关键词

### 环境设置

`export LD_LIBRARY_PATH=/usr/...:/:...:${LD_LIBRARY_PATH}`
一次性有效
`export  PATH=...`
多卡时选择显卡
`export CUDA_VISIBLE_DEVICES=[指定号，0，1，2等]`

## Conan相关

## Shell相关

## GDB使用

## 切面编程
AOP为Aspect Oriented Programming的缩写，意为：面向切面编程，通过预编译方式和运行期间动态代理实现程序功能的统一维护的一种技术。 AOP是OOP的延续，是软件开发中的一个热点，也是Spring框架中的一个重要内容，是函数式编程的一种衍生范型。  
> 面向切面编程（AOP是Aspect Oriented Program的首字母缩写） ，我们知道，面向对象的特点是继承、多态和封装。而封装就要求将功能分散到不同的对象中去，这在软件设计中往往称为职责分配。实际上也就是说，让不同的类设计不同的方法。这样代码就分散到一个个的类中去了。这样做的好处是降低了代码的复杂程度，使类可重用。
但是人们也发现，在分散代码的同时，也增加了代码的重复性。什么意思呢？比如说，我们在两个类中，可能都需要在每个方法中做日志。按面向对象的设计方法，我们就必须在两个类的方法中都加入日志的内容。也许他们是完全相同的，但就是因为面向对象的设计让类与类之间无法联系，而不能将这些重复的代码统一起来。
也许有人会说，那好办啊，我们可以将这段代码写在一个独立的类独立的方法里，然后再在这两个类中调用。但是，这样一来，这两个类跟我们上面提到的独立的类就有耦合了，它的改变会影响这两个类。那么，有没有什么办法，能让我们在需要的时候，随意地加入代码呢？这种在运行时，动态地将代码切入到类的指定方法、指定位置上的编程思想就是面向切面的编程。
一般而言，我们管切入到指定类指定方法的代码片段称为切面，而切入到哪些类、哪些方法则叫切入点。有了AOP，我们就可以把几个类共有的代码，抽取到一个切片中，等到需要时再切入对象中去，从而改变其原有的行为。
这样看来，AOP其实只是OOP的补充而已。OOP从横向上区分出一个个的类来，而AOP则从纵向上向对象中加入特定的代码。有了AOP，OOP变得立体了。如果加上时间维度，AOP使OOP由原来的二维变为三维了，由平面变成立体了。从技术上来说，AOP基本上是通过代理机制实现的。

### python中的切面编程与装饰器
今天来讨论一下装饰器。装饰器是一个很著名的设计模式，经常被用于有切面需求的场景，较为经典的有插入日志、性能测试、事务处理等。装饰器是解决这类问题的绝佳设计，有了装饰器，我们就可以抽离出大量函数中与函数功能本身无关的雷同代码并继续重用。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。  
**内置的装饰器**  
内置的装饰器有三个，分别是staticmethod、classmethod和property  
@property 将某函数，做为属性使用修饰，就是将方法，变成一个属性来使用 
但是这个属性就成为了一个类似c++ private/protect的属性，如果想要修改这个属性的话需要新增一个带有`@属性.setter` 装饰器的方法   
@staticmethod装饰器的功能是去除类的方法默认第一个参数是类的实例，使得该方法成为一个普通的函数，staticmethod是一个类，属于类装饰器  
