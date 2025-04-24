"""
@Author      :Ayaki Shi
@Date        :2025/4/18 11:02 
@Description : 创建模型
"""
from keras import layers, models, optimizers

from config import IMG_SIZE

def create_model():
    model = models.Sequential(
        [
            # 输入层：指定输入数据形状
            layers.Input(shape=(*IMG_SIZE, 3)),
            layers.Rescaling(1./255),  # 归一化到 [0,1]

            # 四层卷积层和四层池化层
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            # 展平层：将输出的多维特征图展平为一维向量
            layers.Flatten(),

            # 防止过拟合
            layers.Dropout(0.5),

            # 两个全连接层，用于特征提取和最终分类
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ]
    )


    # 编译模型
    model.compile(loss='binary_crossentropy',  # 损失函数
                  optimizer= optimizers.Adam(learning_rate=1e-4), # 优化器
                  metrics=['accuracy']) # 评估标准：准确率

    print('--------------构建模型成功--------------')
    return model
