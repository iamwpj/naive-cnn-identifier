#!/usr/bin/env python

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

from pathlib import Path

# Using this guide: https://towardsdatascience.com/build-your-first-cnn-with-tensorflow-a9d7394eaa2e

real_images = [
    file.as_posix() for file in Path("./ffhq-data/images1024x1024").rglob("*.png")
]
synth_images = [file.as_posix() for file in Path("./sfhq-data/images/").rglob("*.jpg")]

np.random.shuffle(real_images)
np.random.shuffle(synth_images)

train_r, val_r, test_r = np.split(
    real_images, [int(len(real_images) * 0.7), int(len(real_images) * 0.8)]
)
train_s, val_s, test_s = np.split(
    synth_images, [int(len(synth_images) * 0.7), int(len(synth_images) * 0.8)]
)

train_real_df = pd.DataFrame({"image": train_r, "label": "real"})
val_real_df = pd.DataFrame({"image": val_r, "label": "real"})
test_real_df = pd.DataFrame({"image": test_r, "label": "real"})

train_synth_df = pd.DataFrame({"image": train_s, "label": "synth"})
val_synth_df = pd.DataFrame({"image": val_s, "label": "synth"})
test_synth_df = pd.DataFrame({"image": test_s, "label": "synth"})

train_df = pd.concat([train_real_df, train_synth_df])
val_df = pd.concat([val_real_df, val_synth_df])
test_df = pd.concat([test_real_df, test_synth_df])

BATCH_SIZE = 32
IMG_HEIGHT = 1024
IMG_WIDTH = 1024

trainGenerator = ImageDataGenerator(rescale=1.0 / 255.0)
valGenerator = ImageDataGenerator(rescale=1.0 / 255.0)
testGenerator = ImageDataGenerator(rescale=1.0 / 255.0)

trainDataset = trainGenerator.flow_from_dataframe(
    dataframe=train_df,
    class_mode="binary",
    x_col="image",
    y_col="label",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)

valDataset = valGenerator.flow_from_dataframe(
    dataframe=val_df,
    class_mode="binary",
    x_col="image",
    y_col="label",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)

testDataset = testGenerator.flow_from_dataframe(
    dataframe=test_df,
    class_mode="binary",
    x_col="image",
    y_col="label",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
)

images, labels = next(iter(testDataset))

# print('Batch shape: ', images.shape)
# print('Label shape: ', labels.shape)
# plt.imshow(images[3])
# print('Label: ', labels[3])


model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(256, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(512, (3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

epochs = 15

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(
    trainDataset,
    epochs=epochs,
    validation_data=(valDataset),
    verbose=0
)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])
plt.savefig("reports/cnn-training-2_accuracy.png")

loss, acc = model.evaluate(testDataset)

print("Loss:", loss)
print("Accuracy:", acc)


# Save model
# https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model

model.save('./saves/cnn-training-2.keras')

