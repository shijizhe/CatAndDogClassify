"""
@Author      :Ayaki Shi
@Date        :2025/4/18 16:08 
@Description : 训练模型
"""

from dataset import create_train_dataset
from model import create_model
from config import EPOCHS, BATCH_SIZE, MODEL_PATH
import matplotlib.pyplot as plt


def train_model():
    # 获取dataset
    train_dataset = create_train_dataset()

    # 生成模型
    model = create_model()

    # 训练模型
    print('--------------开始训练模型--------------')
    history = model.fit(train_dataset,
              epochs = EPOCHS,
              batch_size = BATCH_SIZE)

    # 保存模型
    print('--------------开始保存模型--------------')
    model.save(MODEL_PATH)

    print('--------------开始绘制损失和准确性曲线--------------')
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss', color='blue', marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制训练准确率曲线
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='green', marker='s')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    train_model()



