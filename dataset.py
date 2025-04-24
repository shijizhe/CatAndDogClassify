"""
@Author      :Ayaki Shi
@Date        :2025/4/18 11:02
@Description : 返回dataset
"""

from keras.api.utils import image_dataset_from_directory
from config import train_dir,test2_dir, BATCH_SIZE,IMG_SIZE
from keras import layers, models
import tensorflow as tf

# 数据增强
def create_augmentation_model():
    return models.Sequential([
        layers.RandomFlip("horizontal", seed=42),
        layers.RandomRotation(0.2, fill_mode='nearest', seed=42),
        layers.RandomZoom(0.2, fill_mode='nearest', seed=42),
        layers.RandomContrast(0.3, seed=42),
        layers.RandomTranslation(0.1, 0.1, fill_mode='nearest', seed=42),
    ], name="data_augmentation")

def create_train_dataset():
    train_dataset = image_dataset_from_directory(
        train_dir,
        label_mode = 'binary',
        batch_size = BATCH_SIZE,
        image_size = IMG_SIZE,
        shuffle=True,  # 必须启用 shuffle
        seed=42
    )
    # 创建预处理模型
    augmentation_model = create_augmentation_model()

    # 定义预处理函数
    def preprocess_train(image, label):
        image = augmentation_model(image, training=True)  # 训练模式激活增强
        return image, label

    train_dataset = train_dataset.map(
        preprocess_train,
        num_parallel_calls= tf.data.AUTOTUNE
    )
    print('--------------返回增强后的训练数据集--------------')
    return train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def create_test2_dataset():
    test2_dataset = image_dataset_from_directory(
        test2_dir,
        label_mode = 'binary',
        batch_size = BATCH_SIZE,
        image_size = IMG_SIZE,
        shuffle=False
    )
    print('--------------返回测试数据集[带标签]--------------')
    return test2_dataset
