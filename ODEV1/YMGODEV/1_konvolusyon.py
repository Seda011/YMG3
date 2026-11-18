import tensorflow as tf
import numpy as np

# Rastgelelik kontrolü
tf.keras.utils.set_random_seed(42)

# 3x3 boyutunda 16 filtreli konvolüsyon katmanı
conv = tf.keras.layers.Conv2D(
    filters=8,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer='he_normal'
)

# Örnek giriş görüntüsü
dummy_input = tf.random.normal((1, 128, 128, 1))
# Konvolüsyon işlemi
output = conv(dummy_input)

# Ağırlıkları al
weights, biases = conv.get_weights()

print("Giriş boyutu:", dummy_input.shape)
print("Çıkış boyutu:", output.shape)
print("Filtre ağırlıklarının boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)

# İlk filtrenin ağırlıkları
print("İlk filtre:")
print(weights[:, :, 0, 0])

# Dosyanın en altındaki print işlemlerinden sonra burayı ekle:

'''import matplotlib.pyplot as plt

# 16 filtrenin ilk kanal ağırlıklarını 4x4'lük grid şeklinde yan yana çizdirme
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    # weights dizisinin boyutu (3, 3, 1, 16) şeklindedir
    ax.imshow(weights[:, :, 0, i], cmap='gray')
    ax.axis('off')
    ax.set_title(f"Filtre {i+1}")

plt.tight_layout()
plt.show()'''

import matplotlib.pyplot as plt

# 8 filtrenin ilk kanal ağırlıklarını 4x2'lik grid şeklinde çizdirme
fig, axes = plt.subplots(4, 2, figsize=(6, 8))
for i, ax in enumerate(axes.flat):
    if i < 8:  # Toplam 8 filtre olduğu için sınırı koyuyoruz
        ax.imshow(weights[:, :, 0, i], cmap="gray")
        ax.axis("off")
        ax.set_title(f"Filtre {i+1}")
    else:
        ax.axis("off")  # Boş kalan kareleri gizle

plt.tight_layout()
plt.show()