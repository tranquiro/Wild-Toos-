import base64
import io
import cv2
import keras
import numpy as np
from PIL import Image
from keras.backend import tensorflow_backend as backend
from django.conf import settings
from tensorflow import keras

def detect(upload_image):
    result_name = upload_image.name
    result_list = []
    result_img = ''

    # 設定からカスケードファイルのパスを取得
    cascade_file_path = settings.CASCADE_FILE_PATH
    # 設定からモデルファイルのパスを取得
    model_file_path = settings.MODEL_FILE_PATH
    # kerasでモデルを読み込む
    model = keras.models.load_model(model_file_path, compile=False)
    # アップロードされた画像ファイルをメモリ上でOpenCVのimageに格納
    image = np.asarray(Image.open(upload_image))
    # 画像をOpenCVのBGRからRGB変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 画像をRGBからグレースケール変換
    image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # カスケードファイルの読み込み
    cascade = cv2.CascadeClassifier(cascade_file_path)
    # OpenCVを利用して顔認識
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(64, 64))

    # 顔が１つ以上検出できた場合
    if len(face_list) > 0:
        count = 1
        for (xpos, ypos, width, height) in face_list:
            # 認識した顔の切り抜き
            face_image = image_rgb[ypos:ypos+height, xpos:xpos+width]
            # 切り抜いた顔が小さすぎたらスキップ
            if face_image.shape[0] < 64 or face_image.shape[1] < 64:
                continue
            # 認識した顔のサイズ縮小
            face_image = cv2.resize(face_image, (64, 64))
            # 認識した顔のまわりを赤枠で囲む
            cv2.rectangle(image_rgb, (xpos, ypos),
                          (xpos+width, ypos+height), (0, 0, 255), thickness=2)
            # 認識した顔を1枚の画像を含む配列に変換
            face_image = np.expand_dims(face_image, axis=0)
            # 認識した顔から名前を特定
            name, result = detect_who(model, face_image)
            # 認識した顔に名前を描画
            cv2.putText(image_rgb, f"{count}. {name}", (xpos, ypos+height+20),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            # 結果をリストに格納
            result_list.append(result)
            count = count + 1

    # 画像をPNGに変換
    is_success, img_buffer = cv2.imencode(".png", image_rgb)
    if is_success:
        # 画像をインメモリのバイナリストリームに流し込む
        io_buffer = io.BytesIO(img_buffer)
        # インメモリのバイナリストリームからBASE64エンコードに変換
        result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'", "")

    # tensorflowのバックエンドセッションをクリア
    backend.clear_session()
    # 結果を返却
    return (result_list, result_name, result_img)

def detect_who(model, face_image):
    # 予測
    predicted = model.predict(face_image)
    # 結果
    name = ""
    result = f"オードリー春日の可能性:{predicted[0][0]*100:.2f}% / スギちゃん の可能性:{predicted[0][1]*100:.2f}%"
    name_number_label = np.argmax(predicted)
    if name_number_label == 0:
        name = "Kasuga"
    elif name_number_label == 1:
        name = "Sugityan"
    return (name, result)