import sys
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

filename = sys.argv[1]
print('input:', filename)

# 画像サイズの設定
img_height, img_width = 150, 150

# 分類クラス名の設定、学習時と同じ順番にする
classes = ['horse','zebra']
nb_classes = len(classes)

# 入力画像のロード、4次元テンソルへ変換
img = image.load_img(
        filename,
        target_size = (img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

# 入力データへの正規化
x = x / 255.0

# ファインチューニングしたCNNモデルのロード
model = load_model('finetuning.h5')


# 分類クラスを予測
pred = model.predict(x)[0]

#予測結果を予測確率が上位2件分、クラス名と予測確率を出力
top_n = 2
top_indices = pred.argsort()[-top_n:][::-1]
result = [(classes[i], pred[i]) for i in top_indices]
for x in result:
    print(x)

