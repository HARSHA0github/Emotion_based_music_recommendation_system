# =========================
# 1. Install & Imports
# =========================
!pip install -q tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from google.colab import files

# =========================
# 2. Upload Dataset
# =========================
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]

with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall("dataset")

print("Dataset extracted!")

# =========================
# 3. Config
# =========================
IMG_SIZE = 96   # increased for better feature extraction
BATCH_SIZE = 32
EPOCHS = 25

train_dir = "dataset/train"
test_dir = "dataset/test"

# =========================
# 4. Load Dataset
# =========================
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

class_names = train_data.class_names
print("Classes:", class_names)

# =========================
# 5. Data Augmentation
# =========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.2)
])

# =========================
# 6. Normalize
# =========================
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

train_data = train_data.map(lambda x, y: (preprocess_input(x), y))
test_data = test_data.map(lambda x, y: (preprocess_input(x), y))

# =========================
# 7. Prefetch (Performance)
# =========================
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
test_data = test_data.prefetch(buffer_size=AUTOTUNE)

# =========================
# 8. Build Model (Transfer Learning)
# =========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # freeze base

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 9. Callbacks
# =========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor='val_accuracy',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3
    )
]

# =========================
# 10. Train
# =========================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# 11. Fine-Tuning (Unfreeze top layers)
# =========================
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# =========================
# 12. Plot Accuracy
# =========================
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("Training Accuracy")
plt.show()

# =========================
# 13. Save Model (.h5)
# =========================
model.save("emotion_model.h5")

# =========================
# 14. Save Labels (.npy)
# =========================
np.save("emotion_labels.npy", class_names)

# =========================
# 15. Convert to TFLite
# =========================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("All files saved!")

# =========================
# 16. Download Files
# =========================
files.download("emotion_model.h5")
files.download("emotion_labels.npy")
files.download("emotion_model.tflite")