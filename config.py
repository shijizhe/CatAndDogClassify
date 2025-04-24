"""
@Author      :Ayaki Shi
@Date        :2025/4/18 11:03 
@Description : 配置信息
"""
import os, shutil

data_dir = './data'

# 训练集、测试集所在路径
test_dir = os.path.join(data_dir, 'test')
test2_dir = os.path.join(data_dir, 'test2')
train_dir = os.path.join(data_dir, 'train')

# 划分标签后的数据路径
train_dir_tag_cat = os.path.join(train_dir, 'cats')
test_dir_tag_cat = os.path.join(test_dir, 'cats')
test2_dir_tag_cat = os.path.join(test2_dir, 'cats')

train_dir_tag_dog = os.path.join(train_dir, 'dogs')
test_dir_tag_dog = os.path.join(test_dir, 'dogs')
test2_dir_tag_dog = os.path.join(test2_dir, 'dogs')

# 训练参数
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 15

# 模型路径
MODEL_PATH = './model/CatAndDogClassifier.keras'

"""
# 将train_dir_tag_cat后2500张猫图像移动到test2_dir_tag_cat
cats = ['cat.{}.jpg'.format(i) for i in range(1000)]
for cat in cats:
    src = os.path.join(train_dir_tag_cat, cat)
    dst = os.path.join(test2_dir_tag_cat, cat)
    shutil.move(src, dst)
"""