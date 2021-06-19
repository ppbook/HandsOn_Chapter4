import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# 出力ディレクトリを作成
save_path = 'gene_image'
if os.path.isdir(save_path) == False:
    os.mkdir(save_path)

# 画像ファイル（PIL形式）の読み込み
img = load_img('rena.jpg')

# PIL形式をnumpyのndarray形式に変換
x = img_to_array(img)

# ４次元テンソル形式に変換
x = np.expand_dims(x, 0)

# データ拡張クラスの定義とインスタンスの作成
datagen = ImageDataGenerator(rotation_range = 100)

# 拡張・正規化したデータのジェネレータを生成
gen = datagen.flow(x, batch_size=1, save_to_dir=save_path, save_prefix='image', save_format='jpg')

# 拡張された画像を10枚生成
for i in range(10):
    next(gen)
