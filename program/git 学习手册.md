# git 学习手册

[网址](https://www.bilibili.com/video/BV1pW411A7a5?p=25)

## git结构

> 工作区：写代码
>
> 暂存区：临时存储 git commit $\rightarrow$ 工作区
>
> 本地库：历史版本 git add $\rightarrow$ 暂存区

![img](https://pic1.zhimg.com/80/v2-8e73605a803bd29ceacd86923e08f5f0_720w.jpg)

![img](https://pic2.zhimg.com/80/v2-e95ac3baeb28734107f8bec468525661_720w.jpg)

## git 托管

托管中心 github gitee

操作1：本地库 $\rightarrow$ push $\rightarrow$ 远程库 $\rightarrow$ clone/pull $\rightarrow$ 本地库 

操作2：别人远程库 $\rightarrow$ fork $\rightarrow$ 自己远程库 $\rightarrow$ 自己本地库 $\rightarrow$ push $\rightarrow$ 自己远程库

## GIT命令行操作

### 本地库初始化（建立）

 进入项目文件夹下

`git init` 

### 设置签名

> 用户名:
>
> Email 地址：
>
> 这里设置的与远程库的账号密码无关

命令

1. 项目级别： 当前初始化的文件夹下 本地仓库范围有效

   > `git config user.name ` +用户名
   >
   > `git config user.email `+ 邮箱地址
   >
   > 配置信息保存在 当前项目文件夹下 /.git/config

2. 全局用户级别：登陆当前操作系统的用户范围

   > `git config --global user.name` 
   >
   > `git config --global user.email` 
   >
   > 保存在 ./.gitconfig

3. 级别优先级: 就近原则：项目级别优于系统用户级别(两者均存在)

4. `git config --list`命令来列出Git可以在该处找到的所有的设置
### 添加提交 状态查看

`git status`: 查看工作区、暂存区状态

`git add`+ 文件名 :添加/修改到本地暂存区

`git rm --cached`+ 文件名: 从本地暂存区删除

`git commit [-m] [注释信息]`+文件名: 添加到本地仓库

### 版本记录查看

`git log` :查看提交记录  -[q: 退出 b:上一页 space:下一页]

可选参数:

> git log --pretty=oneline
>
> git log --oneline (仅显示当前版本以及之前的版本)

`git reflog` :在oneline基础上还有一个 `HEAD@{number}`表示移动到当前版本需要的步数，且**显示所有版本**

### 版本前进后退

   > `git reset --[mode]`三种模式：soft、mixed、hard
   >
   > soft:仅在本地库移动HEAD指针
   >
   > mixed:1. 在本地库移动指针 2. 重置暂存区
   >
   > hard:1. 在本地库移动 2. 重置暂存区 3. 重置工作区

1. 基于索引值[推荐]：`git reset --[mode] + [索引值]`(索引值通过`git reflog` 可见)

2. 使用^：往后退一步 `git reset --[mode] HEAD^^`(一个 **^** 退一步 )

3. 使用~: 后退指定步数 `git reset --[mode] HEAD~[number]`(number表示指定的步数)

### 删除文件

前提：删除前 文件存在时的状态提交到本地库

操作： `git reset --hard[指针位置]`

### 比较区别

`git diff [name]` :比较的是工作区与暂存区的区别

`git diff HEAD [name]`：与本地库比较

`git diff[本地库历史版本][name]`

## 分支

### 什么是分支

使用多条线同时推进多个任务，提高开发效率，且保证失败的分支不会破坏整个项目

### 命令

`git branch -v`: 查看分支信息

`git branch [branch_name]`: 创建[branch_name]为分支名的分支

`git checkout [branch_name]`: 切换分支到[branch_name]

**合并分支：**

1. 切换到接受修改的分支  

2. 执行 `git merge [branch_name]`命令，将[branch_name]分支合并到当前分支上

### 克隆远程库到本地库

在本地库项目文件夹下使用git bash`git clone [远程地址]`

### 远程库的拉取

`git pull` 下载并改变本地工作区内容 =`git fetch`+ `git merge` 

`git fetch [远程库地址] [分支]` 仅仅下载并不改变本地工作区内容

可以使用`git checkout `切换到该分支

也可以`git merge` 合并到本地库

 ## vscode 上的使用

默认推送到github，方便平时使用，切换到gitee

### github 使用

### gitee使用

1. git vscode 安装

2. gitee 注册 $\rightarrow$  创建一个仓库(项目)

   >项目url：`https://gitee.com/YourGiteeName/projectname`

3. **法一:**  在本地项目文件夹中打开 **git bash**，使用如下命令

   `git push [url][branch_name]` 完成推送

   此处url 可以存入本地并命名:

   `git remote add [别名][url]` 上述命令可以使用别名进行推送

   `git remote -v` 可以查看远程仓库地址信息

   **法二：** ssh法 
   
   1. `ssh-keygen -t rsa -C "your_e-mail_name@xxx"`
   2. `cat ~/.ssh/id_rsa.pub`
   3. 复制密钥添加到gitee/github
   4. `ssh -T git@gitee.com`



# 综合步骤

> 1.网站创建一个仓库
>
> 2.本地操作 
>
> 2.1 git init
>
> 2.2 
>
> `git config user.name ` +用户名
>
> `git config user.email `+ 邮箱地址 
>
> 2.3 git remote add 【别名】+url
>
> 2.4 git add +文件名
>
> 2.5 git commit
>
> 2.6 git push -u origin[别名] master[分支名] 