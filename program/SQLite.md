# SQLite

SQLite是一个软件库，实现了自给自足的、无服务器的、零配置的、事务性的 SQL 数据库引擎。SQLite是一个增长最快的数据库引擎，这是在普及方面的增长，与它的尺寸大小无关。SQLite 源代码不受版权限制。
SQLite是一个进程内的库，实现了自给自足的、无服务器的、零配置的、事务性的 SQL 数据库引擎。它是一个零配置的数据库，这意味着与其他数据库不一样，您不需要在系统中配置。
就像其他数据库，SQLite 引擎不是一个独立的进程，可以按应用程序需求进行静态或动态连接。SQLite 直接访问其存储文件。

## Why use it?

* 不需要一个单独的服务器进程或操作的系统（无服务器的）。
* SQLite 不需要配置，这意味着不需要安装或管理。
* 一个完整的 SQLite 数据库是存储在一个单一的跨平台的磁盘文件。
* SQLite 是非常小的，是轻量级的，完全配置时小于 400KiB，省略可选功能配置时小于250KiB。
* SQLite 是自给自足的，这意味着不需要任何外部的依赖。
* SQLite 事务是完全兼容 ACID 的，允许从多个进程或线程安全访问。
* SQLite 支持 SQL92（SQL2）标准的大多数查询语言的功能。
* SQLite 使用 ANSI-C 编写的，并提供了简单和易于使用的 API。
* SQLite 可在 UNIX（Linux, Mac OS-X, Android, iOS）和 Windows（Win32, WinCE, WinRT）中运行。

## 基本语法

- 大小写敏感，一般是不区分大小写的，部分命令大小写敏感
- 注释： 注释 `-- comment here` 或者采用c/c++的注释 `/* comment here */`
- 所有语句 `;`结束

## 数据类型

| 存储类  | 描述                 |
| ------- | -------------------- |
| NULL    | 值为NULL             |
| INTEGER | 带符号整数           |
| REAL    | 浮点数               |
| TEXT    | 文本字符串           |
| BLOB    | 完全根据输入进行存储 |

SQLite支持列的亲和类型概念。任何列仍然可以存储任何类型的数据，当数据插入时，该字段的数据将会优先采用亲缘类型作为该值的存储方式

SQLite 没有一个单独的用于存储日期和/或时间的存储类，但 SQLite 能够把日期和时间存储为 TEXT、REAL 或 INTEGER 值。

## command 模式

### 基本命令

```sqlite
.databases --数据库列表
.tables --表的列表
.open [database] --打开数据库
[database] .dump > [name.sql] --数据库保存在一个文本文件中
[database] < [name.sql] --从文本文件中恢复数据库
```

数据库本质上 操作不外乎 **增删改查**
数据定义（表头）

| command | description                                            |
| ------- | ------------------------------------------------------ |
| CREATE  | 创建一个新的表，一个表的视图，或者数据库中的其他对象。 |
| ALTER   | 修改数据库中的某个已有的数据库对象，比如一个表。       |
| DROP    | 删除整个表，或者表的视图，或者数据库中的其他对象。     |

```
CREATE TABLE database_name.table_name(
   column1 datatype  PRIMARY KEY(one or more columns),
   column2 datatype,
   column3 datatype,
   .....
   columnN datatype,
);

/*e.g.*/
sqlite> CREATE TABLE COMPANY(
   ID INT PRIMARY KEY     NOT NULL,
   NAME           TEXT    NOT NULL,
   AGE            INT     NOT NULL,
   ADDRESS        CHAR(50),
   SALARY         REAL
);
```
```
DROP TABLE database_name.table_name;
```

```
INSERT INTO TABLE_NAME [(column1, column2, column3,...columnN)]  
VALUES (value1, value2, value3,...valueN);
-- 或者
INSERT INTO TABLE_NAME VALUES (value1,value2,value3,...valueN);

-- e.g.
INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)
VALUES (1, 'Paul', 32, 'California', 20000.00 );
INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)
VALUES (2, 'Allen', 25, 'Texas', 15000.00 );
INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)
VALUES (3, 'Teddy', 23, 'Norway', 20000.00 );
-- 另一种
INSERT INTO COMPANY VALUES (7, 'James', 24, 'Houston', 10000.00 );
```




数据操作+查询（表内容）

| command | description                    |
| ------- | ------------------------------ |
| INSERT  | 创建/插入一条记录              |
| UPDATE  | 修改记录                       |
| DELETE  | 删除记录                       |
| SELECT  | 从一个或多个表中检索某些记录。 |

## Python api
```python
import sqlite3 as sql
# 必须创建一个Connection对象，用于与数据库文件通信(读写)
conn = sql.connect('name.db')
#或者使用 :memory: 在内存创建一个空数据库
#对Connection 对象创建Cursor游标对象，然后就可以调用execute()方法执行SQL语句了
c = conn.cursor()

# Create table
c.execute('''create table table_name (id int, name text,...)''')
# Insert a Instance(row) 
c.execute('''insert into table_name values (....)''')

#save the changes
conn.commit()

#end
conn.close()
```

## C/C++ api
