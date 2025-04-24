"""
@Author      :Ayaki Shi
@Date        :2025/4/18 16:45 
@Description : 测试模型
"""
from keras import models
import numpy as np
import os, shutil
from keras_preprocessing import image

from config import MODEL_PATH,IMG_SIZE,test_dir,test_dir_tag_cat,test_dir_tag_dog



DOG_TAG_STR = 'dog'
CAT_TAG_STR = 'cat'
NUM_IMAGES = 12500             # 测试图片数

class CatAndDogClassifier:
    def __init__(self):
        self.model = models.load_model(MODEL_PATH)
        print("模型加载成功！")

    def predict_single_image(self, img_path):
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        # 错误代码：双重归一化
        # img_array = np.expand_dims(img_array, axis=0) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)[0][0]
        print(prediction)
        return DOG_TAG_STR if prediction > 0.5 else CAT_TAG_STR, prediction


    def classify_all_images(self):
        # 遍历所有图片
        # for i in range(1, NUM_IMAGES + 1):
        filename = ''
        for i in range(1, NUM_IMAGES + 1):
            try:
                #（文件名为1.jpg到12500.jpg）
                filename = f"{i}.jpg"
                src_path = os.path.join(test_dir, filename)

                # 跳过不存在的文件
                if not os.path.exists(src_path):
                    print(f"Warning: {filename} 不存在，已跳过")
                    continue

                # 进行预测
                label, confidence = self.predict_single_image(src_path)

                # 确定目标目录
                dest_dir = test_dir_tag_dog if label == DOG_TAG_STR else test_dir_tag_cat
                dest_path = os.path.join(dest_dir, filename)

                # 移动文件
                shutil.move(src_path, dest_path)

                if i%500 == 0: # 打印12500行太多了，每500行打印一次
                    print(f"[{i}/12500] {filename} -> {dest_dir} (置信度: {confidence:.2%})")

            except Exception as e:
                print(f"处理 {filename} 时发生错误: {str(e)}")
                continue


def evaluate_model():
    from dataset import create_test2_dataset

    test2_dataset = create_test2_dataset()

    model = models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(test2_dataset)
    print(f'\nTest accuracy: {acc:.2%}')


if __name__ == '__main__':
    # 初始化分类器
    classifier = CatAndDogClassifier()

    # # 评估整体准确率
    # evaluate_model()


    # # 单张图片预测
    # img_path = os.path.join('./data/train/dogs/dog.100.jpg')
    # label, prob = classifier.predict_single_image(img_path)
    # print(f'预测为: {label} (置信度: {prob if label == DOG_TAG_STR else 1 - prob:.2%})')

    # 将不带标签的测试图片分类放入不同的文件夹
    classifier.classify_all_images()
