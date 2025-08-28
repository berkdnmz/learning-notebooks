import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from scipy import ndimage

# Verileri yükleme
X_train = np.loadtxt("./data/input.csv", delimiter=",")
Y_train = np.loadtxt("./data/labels.csv", delimiter=",")
X_test = np.loadtxt("./data/input_test.csv", delimiter=",")
Y_test = np.loadtxt("./data/labels_test.csv", delimiter=",")

# Verileri yeniden şekillendirme
X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)
X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

# Normalizasyon
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Eğitim verisi boyutu: {X_train.shape}")
print(f"Test verisi boyutu: {X_test.shape}")
print(f"Kedi sayısı: {np.sum(Y_train == 1)}, Köpek sayısı: {np.sum(Y_train == 0)}")


# 1. YÖNTEM: ImageDataGenerator ile veri çoğaltma
def augment_data_with_generator(X_train, Y_train, augment_factor=5):
    """ImageDataGenerator ile veri çoğaltma"""

    # Sadece kedi örneklerini seç (Y_train == 1)
    cat_indices = np.where(Y_train == 1)[0]
    X_cats = X_train[cat_indices]
    Y_cats = Y_train[cat_indices]

    # Veri çoğaltma için generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_cats = []
    augmented_labels = []

    print("Veri çoğaltma başlıyor...")

    for i in range(len(X_cats)):
        img = X_cats[i].reshape(1, 100, 100, 3)

        # Her kedi görüntüsünden yeni örnekler oluştur
        count = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_cats.append(batch[0])
            augmented_labels.append(1)
            count += 1

            if count >= augment_factor:
                break

    # Yeni verileri orijinal veriye ekle
    X_train_augmented = np.concatenate([X_train, np.array(augmented_cats)])
    Y_train_augmented = np.concatenate([Y_train, np.array(augmented_labels).reshape(-1, 1)])

    print(f"Orijinal veri: {len(X_train)}")
    print(f"Çoğaltılan kedi örnekleri: {len(augmented_cats)}")
    print(f"Toplam veri: {len(X_train_augmented)}")

    return X_train_augmented, Y_train_augmented


# 2. YÖNTEM: Manuel veri çoğaltma
def augment_data_manually(X_train, Y_train):
    """Manuel olarak veri çoğaltma"""

    cat_indices = np.where(Y_train == 1)[0]
    X_cats = X_train[cat_indices]

    augmented_cats = []

    for image in X_cats:
        # Orijinal
        augmented_cats.append(image)

        # Yatay çevirme
        augmented_cats.append(np.fliplr(image))

        # 90 derece döndürme
        augmented_cats.append(ndimage.rotate(image, 90, reshape=False))

        # 180 derece döndürme
        augmented_cats.append(ndimage.rotate(image, 180, reshape=False))

        # Parlaklık artırma
        brightened = np.clip(image * 1.3, 0, 1)
        augmented_cats.append(brightened)

        # Parlaklık azaltma
        darkened = np.clip(image * 0.7, 0, 1)
        augmented_cats.append(darkened)

    augmented_labels = [1] * len(augmented_cats)

    X_train_augmented = np.concatenate([X_train, np.array(augmented_cats)])
    Y_train_augmented = np.concatenate([Y_train, np.array(augmented_labels).reshape(-1, 1)])

    print(f"Manuel çoğaltma: {len(augmented_cats)} yeni kedi örneği eklendi")

    return X_train_augmented, Y_train_augmented


# 3. YÖNTEM: Class weight ile dengeleme
def train_with_class_weights(X_train, Y_train, X_test, Y_test):
    """Class weight kullanarak eğitim"""

    # Class weight hesapla
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(Y_train.flatten()),
        y=Y_train.flatten()
    )
    class_weights = dict(enumerate(class_weights))

    print(f"Class weights: {class_weights}")

    # Model oluştur
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(100, 100, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Modeli eğit
    history = model.fit(X_train, Y_train, epochs=10, batch_size=64,
                        class_weight=class_weights, validation_data=(X_test, Y_test))

    return model, history


# 4. YÖNTEM: Real-time data augmentation
def train_with_realtime_augmentation(X_train, Y_train, X_test, Y_test):
    """Real-time data augmentation ile eğitim"""

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow(
        X_train, Y_train,
        batch_size=64,
        subset='training'
    )

    validation_generator = train_datagen.flow(
        X_train, Y_train,
        batch_size=32,
        subset='validation'
    )

    # Model oluştur
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(100, 100, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Modeli eğit
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )

    return model, history


# Model değerlendirme fonksiyonu
def evaluate_model(model, X_test, Y_test):
    """Modeli değerlendir ve örnek tahminler göster"""

    # Model değerlendirme
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Kaybı: {loss:.4f}")
    print(f"Test Doğruluğu: {accuracy:.4f}")

    # Rastgele örneklerle tahmin
    plt.figure(figsize=(15, 10))

    for i in range(6):
        idx = random.randint(0, len(Y_test) - 1)

        plt.subplot(2, 3, i + 1)
        plt.imshow(X_test[idx, :])

        y_pred = model.predict(X_test[idx, :].reshape(1, 100, 100, 3))
        y_pred = y_pred > 0.5

        actual = "cat" if Y_test[idx] == 1 else "dog"
        predicted = "cat" if y_pred == 1 else "dog"

        color = 'green' if actual == predicted else 'red'

        plt.title(f"Gerçek: {actual}\nTahmin: {predicted}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Ana çalıştırma kısmı
def main():
    print("Kedi-Köpek Sınıflandırma Modeli")
    print("=" * 40)

    # Hangi yöntemi kullanmak istediğinizi seçin
    print("1. ImageDataGenerator ile veri çoğaltma")
    print("2. Manuel veri çoğaltma")
    print("3. Class weight ile eğitim")
    print("4. Real-time data augmentation")

    choice = input("Lütfen bir yöntem seçin (1-4): ")

    if choice == "1":
        print("ImageDataGenerator ile veri çoğaltma seçildi...")
        X_train_aug, Y_train_aug = augment_data_with_generator(X_train, Y_train)

        # Model oluştur ve eğit
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(100, 100, 3)))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X_train_aug, Y_train_aug, epochs=10, batch_size=64, validation_data=(X_test, Y_test))

        evaluate_model(model, X_test, Y_test)

    elif choice == "2":
        print("Manuel veri çoğaltma seçildi...")
        X_train_aug, Y_train_aug = augment_data_manually(X_train, Y_train)

        # Model oluştur ve eğit
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(100, 100, 3)))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X_train_aug, Y_train_aug, epochs=10, batch_size=64, validation_data=(X_test, Y_test))

        evaluate_model(model, X_test, Y_test)

    elif choice == "3":
        print("Class weight ile eğitim seçildi...")
        model, history = train_with_class_weights(X_train, Y_train, X_test, Y_test)
        evaluate_model(model, X_test, Y_test)

    elif choice == "4":
        print("Real-time data augmentation seçildi...")
        model, history = train_with_realtime_augmentation(X_train, Y_train, X_test, Y_test)
        evaluate_model(model, X_test, Y_test)

    else:
        print("Geçersiz seçim. Orijinal veri ile eğitim yapılıyor...")

        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(100, 100, 3)))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test))

        evaluate_model(model, X_test, Y_test)


# Programı çalıştır
if __name__ == "__main__":
    main()