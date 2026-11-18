import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)
# Normalizasyon
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# Kanal boyutu ekleme
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    # Birinci konvolüsyon
    tf.keras.layers.Conv2D(
        8, (3, 3),
        activation='relu'
    ),
    # Birinci pooling
    tf.keras.layers.MaxPooling2D((2, 2)),
    # İkinci konvolüsyon
    tf.keras.layers.Conv2D(
        16, (3, 3),
        activation='relu'
    ),
    # İkinci pooling
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Çok boyutlu özellikleri tek boyuta dönüştür
    tf.keras.layers.Flatten(),
    # Tam bağlantılı katman
    tf.keras.layers.Dense(64, activation='relu'),
    # 10 sınıf için çıktı
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()