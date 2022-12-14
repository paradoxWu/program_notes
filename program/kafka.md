# kafka

首先明确一个应用场景，kafka是一个消息中间件框架api。
在分布式系统中完成消息的发送和接收的基础软件。消息中间件也可以称消息队列，是指用高效可靠的消息传递机制进行与平台无关的数据交流，并基于数据通信来进行分布式系统的集成。通过提供消息传递和消息队列模型，可以在分布式环境下扩展进程的通信。

产生消息中间件的原因，个人理解有这个几点:

1. 解耦： 消息的发送方与接收方不是一对一的状态，绑定这里的发送方(生产者)与接收方(消费者)会出现问题
2. 异步: 解耦后就可以做到每一方专注于自己任务，生产者只管生产消息，消费者只管接收消息，这中间由消息中间件完成，不会出现生产者生产完等待消费者消费之后再进行下一次生产的情况，提高效率。
3. 削峰

## 模式

### 点对点

点对点模式通常是基于拉取或者轮询的消息传送模型，这个模型的特点是发送到队列的消息被一个且只有一个消费者进行处理。生产者将消息放入消息队列后，由消费者主动的去拉取消息进行消费。点对点模型的的优点是消费者拉取消息的频率可以由自己控制。但是消息队列是否有消息需要消费，在消费者端无法感知，所以在消费者端需要额外的线程去监控

### 发布订阅

布订阅模式是一个基于消息送的消息传送模型，改模型可以有多种不同的订阅者。生产者将消息放入消息队列后，队列会将消息推送给订阅过该类消息的消费者（类似微信公众号）。由于是消费者被动接收推送，所以无需感知消息队列是否有待消费的消息！但是consumer1、consumer2、consumer3由于机器性能不一样，所以处理消息的能力也会不一样，但消息队列却无法感知消费者消费的速度！所以推送的速度成了发布订阅模模式的一个问题！假设三个消费者处理速度分别是8M/s、5M/s、2M/s，如果队列推送的速度为5M/s，则consumer3无法承受！如果队列推送的速度为2M/s，则consumer1、consumer2会出现资源的极大浪费！

## kafka概念

Producer：Producer即生产者，消息的产生者，是消息的入口。
在Kafka中，客户端和服务器之间的通信是通过简单，高性能，语言无关的TCP协议完成的

kafka cluster：

Broker：Broker是kafka实例，每个服务器上有一个或多个kafka的实例，我们姑且认为每个broker对应一台服务器。每个kafka集群内的broker都有一个不重复的编号，如图中的broker-0、broker-1等……

Topic：消息的主题，可以理解为消息的分类，kafka的数据就保存在topic。在每个broker上都可以创建多个topic。

Partition：Topic的分区，每个topic可以有多个分区，分区的作用是做负载，提高kafka的吞吐量。同一个topic在不同的分区的数据是不重复的，partition的表现形式就是一个一个的文件夹！

Replication:每一个分区都有多个副本，副本的作用是做备胎。当主分区（Leader）故障的时候会选择一个备胎（Follower）上位，成为Leader。在kafka中默认副本的最大数量是10个，且副本的数量不能大于Broker的数量，follower和leader绝对是在不同的机器，同一机器对同一个分区也只可能存放一个副本（包括自己）。

Message：每一条发送的消息主体。

Consumer：消费者，即消息的消费方，是消息的出口。

Consumer Group：我们可以将多个消费组组成一个消费者组，在kafka的设计中同一个分区的数据只能被消费者组中的某一个消费者消费。同一个消费者组的消费者可以消费同一个topic的不同分区的数据，这也是为了提高kafka的吞吐量！

Zookeeper：kafka集群依赖zookeeper来保存集群的的元信息，来保证系统的可用性。

## 存储策略

无论消息是否被消费，kafka都会保存所有的消息。那对于旧数据有什么删除策略呢？

1、 基于时间，默认配置是168小时（7天）。

2、 基于大小，默认配置是1073741824。

需要注意的是，kafka读取特定消息的时间复杂度是O(1)，所以这里删除过期的文件并不会提高kafka的性能！

## 消费数据

消息存储在log文件后，消费者就可以进行消费了。在讲消息队列通信的两种模式的时候讲到过点对点模式和发布订阅模式。Kafka采用的是发布订阅的模式，消费者主动的去kafka集群拉取消息，与producer相同的是，消费者在拉取消息的时候也是找leader去拉取。

多个消费者可以组成一个消费者组（consumer group），每个消费者组都有一个组id！同一个消费组者的消费者可以消费同一topic下不同分区的数据，但是不会组内多个消费者消费同一分区的数据！！！
**在实际的应用中，建议消费者组的consumer的数量与partition的数量一致！**

## 核心api

1. Producer API允许应用程序发布记录流至一个或多个kafka的topics（主题）
2. Consumer API允许应用程序订阅一个或多个topics（主题），并处理所产生的对他们记录的数据流。
3. Streams API允许应用程序充当流处理器，从一个或多个topics（主题）消耗的输入流，并产生一个输出流至一个或多个输出的topics（主题），有效地变换所述输入流，以输出流
4. Connector API
   允许构建和运行kafka topics（主题）连接到现有的应用程序或数据系统中重用生产者或消费者

## kafka文件

解压完kafka下载包后文件目录如下:

> LICENSE
> NOTICE
> bin/
> config/
> libs/
> licenses/
> site-docs/

### bin/
启动各种服务的sh脚本
### config/
下面有 consumer.properties, producer.properties, server.properties 
需要修改一些参数，主要修改broker.id，listeners，log.dirs，zookeeper.connect四个参数 
配置server.properties  
>broker.id = 0
>log dirs = xx/xxx/
>zookeeper.connect=localhost:port_num 可以修改为其他
### lib/
第三方相关库

