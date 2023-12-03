#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
from PIL import Image,ImageOps
from pathlib import Path

IMG_HEIGHT = 1024
IMG_WIDTH = 1024

model = keras.models.load_model('saves/cnn-training-2.keras')

def standardize_size(image):
    smaller_dim = image.width if image.width <= image.height else image.height

    # Make no changes to "square-ness" unless we need to
    if smaller_dim != (image.width + image.height) / 2:
        left = int(image.width - smaller_dim) / 2
        top = int(image.height - smaller_dim) / 2
        right = left
        bottom = top

        squared = ImageOps.crop(
            image,
            border=(
                left,
                top,
                right,
                bottom,
            ),
        )
    else:
        squared = image

    result = squared.resize((IMG_HEIGHT, IMG_WIDTH))
    result.save(f"predict_modified/{Path(image.filename).name}")
    return result

def preprocess(image):
    img_resize = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    img_norm = img_resize / 255
    return img_norm

def prediction(images=[]):
    predictions = {}
    for image in images:
        print(f"{image.name=}")
        raw = Image.open(image)

        if raw.mode != 'RGB':
            raw = raw.convert('RGB')

        raw_standard = standardize_size(raw)
        # raw_standard = raw
        img = tf.reshape(raw_standard, (-1, IMG_HEIGHT, IMG_HEIGHT, 3))
        img = preprocess(img)
        guess = model.predict(img)

        predictions[image.name] = {
            'prediction': 'real' if round(guess[0][0]) == 0 else 'synthetic',
            'exact_prediction':  f'{guess[0][0]:.10f}',
            'source': image.parent.as_posix()
        }

    
    return predictions

if __name__ == "__main__":
    print("Please call this by importing and providing `prediction()` with a list of image files.")