from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

# 画像サイズとディレクトリの設定
img_width, img_height = 150, 150
train_data_dir = 'data/train'
test_data_dir = 'data/test'

# エポック数の設定
epoch = 20

# 分類クラス名の設定
classes = ['horse','zebra']
nb_classes = len(classes)

# VGG16モデルのロード
vgg_model = VGG16(
        include_top = False,
        weights = 'imagenet',
        input_shape = (img_height, img_width, 3))

# VGG16モデルの下に全結合層を追加
model = Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

# VGG16モデルの上位15層のパラメータを凍結
for layer in vgg_model.layers[:15]:
    layer.trainable = False

model.summary()

# 最適化関数のパラメータ設定
sgd = optimizers.SGD(lr = 0.001, momentum = 0.1, decay = 0.0)

# 損失関数は交差エントロピー、最適化関数は確率的勾配法
model.compile(
        loss = 'categorical_crossentropy',
        optimizer = sgd,
        metrics = ['accuracy'])

# 学習データのデータ拡張を設定
train_datagen = ImageDataGenerator(
        rescale = 1.0 / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

# 評価データのデータ拡張を設定
test_datagen = ImageDataGenerator(rescale = 1.0 / 255)

# 学習データのジェネレータを生成
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        classes = classes,
        batch_size = 32,
        class_mode = 'categorical')

# 評価データのジェネレータを生成
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_height, img_width),
        classes = classes,
        batch_size = 32,
        class_mode = 'categorical')

# コールバック関数（モデルの保存）の設定
mc_cb = ModelCheckpoint(
        filepath = 'finetuning.h5',
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True)

# ジェネレータを用いたモデルの学習
history = model.fit_generator(
    train_generator,
    epochs=epoch,
    validation_data = test_generator,
    callbacks = [mc_cb])

#予測誤差のグラフ化
plt.plot(range(len(history.history['loss'])), history.history['loss'], marker='o', color = 'black', label='loss')
plt.plot(range(len(history.history['val_loss'])), history.history['val_loss'], marker='v', linestyle='--', color='black', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

#予測精度のグラフ化
plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'], marker='o', color = 'black', label='acc')
plt.plot(range(len(history.history['val_accuracy'])), history.history['val_accuracy'], marker='v', linestyle='--', color = 'black', label='val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

