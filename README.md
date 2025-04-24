基于Keras3.x使用CNN实现简单的猫狗分类，置信度约为：85%

> 详细项目介绍见博客：[基于Keras3.x使用CNN实现简单的猫狗分类，置信度约为：85%](https://blog.csdn.net/shijizhe1/article/details/147482306)
> 

# 环境版本
* python 3.11
* keras 3.9.2
* tensorflow 2.19.0
# 模型结构
* 输入层：指定输入数据形状
* 数据归一化
* 四层卷积层和四层池化层交替
* 展平层：将输出的多维特征图展平为一维向量
* Dropout防止过拟合
* 两个全连接层，用于特征提取和最终分类
# 项目目录
* /data 存放数据集
* /model 存放训练好的模型
* /config.py 存储一些关键模型参数和路径信息等
* /dataset.py 返回数据增强后的数据集，用于模型训练
* /model.py 定义模型
* /train.py  训练模型并绘制训练损失和准确度曲线
* /test.py 测试模型精准度