from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from matplotlib import pyplot as plt

# バッチサイズ、分類クラス数、エポック数の設定
batch_size = 128
num_classes = 10
epochs = 20

# MINISTデータセットの読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 4次元テンソル形式に変換
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# データの正規化および実数値化
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# ラベルデータをOne-hotベクトルに変換
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#CNNモデルの定義
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

#CNNモデルの可視化
model.summary()

#CNNモデルのコンパイル
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#CNNモデルの学習
history = model.ﬁt(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# CNNモデルの予測精度
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

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
