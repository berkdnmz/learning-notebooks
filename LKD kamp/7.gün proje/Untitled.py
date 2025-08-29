şimdi en son buydu kod ona göre entegre et # -------------------------
# 1️⃣ Gerekli Kütüphaneler
# -------------------------
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Pickle ile CIFAR-100 yükleme fonksiyonu
import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# -------------------------
# 2️⃣ Veri Yükleme ve Ön İşleme
# -------------------------
# Dosya yolları (PC'deki 'data' klasörüne göre)
from google.colab import drive
drive.mount('/content/drive')
import os
data_train_path = './drive/MyDrive/data/train'
data_test_path  = './drive/MyDrive/data/test'
data_meta_path  = './drive/MyDrive/data/meta'

# Veriyi yükle
train_dict = unpickle(data_train_path)
test_dict  = unpickle(data_test_path)
meta_dict  = unpickle(data_meta_path)

# 100 alt sınıf ve 20 süper sınıf isimleri
fine_labels   = [x.decode('utf-8') for x in meta_dict[b'fine_label_names']]
coarse_labels = [x.decode('utf-8') for x in meta_dict[b'coarse_label_names']]

# Train / Test verisi
X_train_original = train_dict[b'data']
y_train_original = np.array(train_dict[b'fine_labels'])
X_test  = test_dict[b'data']
y_test  = np.array(test_dict[b'fine_labels'])

# Train ve valid split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_original, y_train_original, train_size=0.8, stratify=y_train_original, random_state=42
)

# Reshape ve normalize
def reshape_and_normalize(X):
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X / 255.0

X_train = reshape_and_normalize(X_train)
X_valid = reshape_and_normalize(X_valid)
X_test  = reshape_and_normalize(X_test)

# One-hot encoding
y_train_cat = to_categorical(y_train, 100)
y_valid_cat = to_categorical(y_valid, 100)
y_test_cat  = to_categorical(y_test, 100)

# -------------------------
# 3️⃣ KÖTÜ PERFORMANS GÖSTEREN SINIFLAR İÇİN ÖZEL İYİLEŞTİRME
# -------------------------

# Önceki sonuçlara göre en kötü 15 sınıf
worst_15_classes = [
    'otter', 'seal', 'squirrel', 'rabbit', 'man', 'mouse', 
    'beaver', 'bear', 'shrew', 'woman', 'lobster', 'crocodile', 
    'turtle', 'possum', 'tulip'
]

print("🎯 Kötü Performanslı Sınıflar için Özel İyileştirme Başlıyor...")
print(f"📉 İyileştirilecek sınıflar: {worst_15_classes}")

# -------------------------
# 4️⃣ BU SINIFLAR İÇİN ÖZEL VERİ ARTIRMA
# -------------------------

def enhanced_augmentation_for_worst_classes(images, labels, class_names, enhancement_factor=3):
    """
    Kötü performans gösteren sınıflar için özel veri artırma
    """
    enhanced_images = []
    enhanced_labels = []
    
    # Orijinal veriyi ekle
    enhanced_images.extend(images)
    enhanced_labels.extend(labels)
    
    # Her kötü sınıf için özel artırma
    for class_name in class_names:
        class_idx = fine_labels.index(class_name)
        class_mask = (labels == class_idx)
        class_images = images[class_mask]
        
        if len(class_images) > 0:
            print(f"🔍 {class_name} sınıfı için veri artırma ({len(class_images)} örnek)")
            
            # Bu sınıf için enhancement_factor katı kadar yeni örnek oluştur
            for _ in range(enhancement_factor):
                augmented_batch = []
                
                for img in class_images:
                    # Çeşitli augmentasyonlar
                    augmented = img.copy()
                    
                    # Rastgele dönüşümler
                    if np.random.random() > 0.5:
                        augmented = tf.image.random_flip_left_right(augmented[np.newaxis, ...])[0]
                    if np.random.random() > 0.5:
                        augmented = tf.image.random_flip_up_down(augmented[np.newaxis, ...])[0]
                    if np.random.random() > 0.5:
                        augmented = tf.image.random_brightness(augmented, 0.2)
                    if np.random.random() > 0.5:
                        augmented = tf.image.random_contrast(augmented, 0.8, 1.2)
                    
                    # Küçük rotasyon ve zoom
                    if np.random.random() > 0.5:
                        rotation_layer = tf.keras.layers.RandomRotation(0.1)
                        augmented = rotation_layer(augmented[np.newaxis, ...])[0]
                    
                    augmented_batch.append(augmented)
                
                # Artırılmış veriyi ekle
                enhanced_images.extend(augmented_batch)
                enhanced_labels.extend([class_idx] * len(augmented_batch))
    
    return np.array(enhanced_images), np.array(enhanced_labels)

# Kötü sınıflar için veri artırma uygula
print("🔄 Kötü sınıflar için veri artırma uygulanıyor...")
X_train_enhanced, y_train_enhanced = enhanced_augmentation_for_worst_classes(
    X_train, y_train, worst_15_classes, enhancement_factor=2
)

print(f"📊 Veri boyutu: {len(X_train)} -> {len(X_train_enhanced)}")
print(f"📈 Artış oranı: {len(X_train_enhanced)/len(X_train):.2f}x")

# One-hot encoding'i yenile
y_train_enhanced_cat = to_categorical(y_train_enhanced, 100)

# -------------------------
# 5️⃣ SINIF AĞIRLIKLARINI GÜNCELLE
# -------------------------

# Güncellenmiş sınıf ağırlıkları
class_weights_enhanced = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_enhanced),
    y=y_train_enhanced
)
class_weight_dict_enhanced = dict(enumerate(class_weights_enhanced))

# Kötü sınıfların ağırlıklarını biraz daha artır
for class_name in worst_15_classes:
    class_idx = fine_labels.index(class_name)
    if class_idx in class_weight_dict_enhanced:
        class_weight_dict_enhanced[class_idx] *= 1.5  # %50 daha fazla ağırlık

print("⚖️  Sınıf ağırlıkları güncellendi (kötü sınıflar +%50 ağırlık)")

# -------------------------
# 6️⃣ FOCUSED MODEL MİMARİSİ
# -------------------------

def create_focused_model():
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Hafif veri artırma
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    
    # 1. Blok
    x = layers.Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(96, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)
    
    # 2. Blok
    x = layers.Conv2D(192, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(192, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.35)(x)
    
    # 3. Blok
    x = layers.Conv2D(384, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(384, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.45)(x)
    
    # 4. Blok
    x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Daha geniş fully connected layers
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Çıkış katmanı
    outputs = layers.Dense(100, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# -------------------------
# 7️⃣ ODAKLANMIŞ EĞİTİM
# -------------------------

# Modeli oluştur
focused_model = create_focused_model()

# Modeli derle
focused_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0008),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    mode='max'
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="focused_model_checkpoints/best_model.keras",
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    mode='max',
    verbose=1
)

# Dataset oluştur
train_dataset_enhanced = tf.data.Dataset.from_tensor_slices((X_train_enhanced, y_train_enhanced_cat))
train_dataset_enhanced = train_dataset_enhanced.shuffle(15000).batch(128).prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid_cat))
valid_dataset = valid_dataset.batch(128).prefetch(tf.data.AUTOTUNE)

# Modeli eğit
print("🎯 Odaklanmış model eğitimi başlıyor...")
history_focused = focused_model.fit(
    train_dataset_enhanced,
    epochs=120,
    validation_data=valid_dataset,
    class_weight=class_weight_dict_enhanced,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# -------------------------
# 8️⃣ DETAYLI PERFORMANS ANALİZİ
# -------------------------

# Test değerlendirmesi
print("📊 Detaylı performans analizi yapılıyor...")
test_loss, test_accuracy = focused_model.evaluate(X_test, y_test_cat, verbose=1)
print(f"🎯 Yeni Test Doğruluğu: {test_accuracy:.4f}")

# Tahminler
y_pred = focused_model.predict(X_test, verbose=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Tüm sınıflar için detaylı analiz
def detailed_class_analysis(true_labels, pred_labels, class_names):
    results = []
    for i, class_name in enumerate(class_names):
        class_mask = true_labels == i
        class_correct = np.sum((pred_labels == i) & class_mask)
        class_total = np.sum(class_mask)
        accuracy = class_correct / class_total if class_total > 0 else 0
        
        # Yanlış tahminlerin dağılımı
        wrong_mask = (pred_labels != i) & class_mask
        wrong_predictions = pred_labels[wrong_mask]
        
        if len(wrong_predictions) > 0:
            wrong_counts = np.bincount(wrong_predictions, minlength=len(class_names))
            top_3_wrong = wrong_counts.argsort()[-3:][::-1]
            top_3_info = []
            
            for wrong_idx in top_3_wrong:
                if wrong_counts[wrong_idx] > 0:
                    top_3_info.append(f"{class_names[wrong_idx]}:{wrong_counts[wrong_idx]}")
        else:
            top_3_info = ["None"]
            
        results.append({
            'class': class_name,
            'accuracy': accuracy,
            'total_samples': class_total,
            'top_3_wrong': ", ".join(top_3_info)
        })
    
    return pd.DataFrame(results)
# Detaylı analiz
detailed_results = detailed_class_analysis(y_test, y_pred_labels, fine_labels)
detailed_results = detailed_results.sort_values('accuracy')

# Kötü sınıfların gelişimini analiz et
print("\n📈 KÖTÜ SINIFLARIN GELİŞİM ANALİZİ:")
print("=" * 60)

worst_classes_analysis = detailed_results[detailed_results['class'].isin(worst_15_classes)]
print(worst_classes_analysis[['class', 'accuracy', 'top_3_wrong']].to_string(index=False))

# İyileşme oranlarını hesapla (önceki sonuçlara göre)
previous_accuracies = {
    'otter': 0.15, 'seal': 0.15, 'squirrel': 0.21, 'rabbit': 0.22,
    'man': 0.24, 'mouse': 0.20, 'beaver': 0.23, 'bear': 0.18,
    'shrew': 0.33, 'woman': 0.24, 'lobster': 0.42, 'crocodile': 0.51,
    'turtle': 0.39, 'possum': 0.27, 'tulip': 0.37
}

improvement_data = []
for _, row in worst_classes_analysis.iterrows():
    class_name = row['class']
    new_accuracy = row['accuracy']
    old_accuracy = previous_accuracies.get(class_name, 0.0)
    improvement = new_accuracy - old_accuracy
    improvement_pct = (improvement / old_accuracy) * 100 if old_accuracy > 0 else 0
    
    improvement_data.append({
        'class': class_name,
        'old_accuracy': old_accuracy,
        'new_accuracy': new_accuracy,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    })

improvement_df = pd.DataFrame(improvement_data)
improvement_df = improvement_df.sort_values('improvement_pct', ascending=False)

print("\n📊 İYİLEŞME RAPORU:")
print("=" * 60)
print(improvement_df.to_string(index=False))

# Genel sınıflandırma raporu
print("\n📋 GENEL SINIFLANDIRMA RAPORU:")
print("=" * 60)
print(classification_report(y_test, y_pred_labels, target_names=fine_labels, digits=4))

# -------------------------
# 9️⃣ GÖRSELLEŞTİRME VE KAYDETME
# -------------------------

# İyileşme grafiği
plt.figure(figsize=(12, 8))
plt.barh(improvement_df['class'], improvement_df['improvement_pct'])
plt.xlabel('İyileşme Yüzdesi (%)')
plt.title('Kötü Sınıflarda İyileşme Oranları')
plt.tight_layout()
plt.savefig('class_improvement_analysis.png')
plt.show()

# Modeli kaydet
focused_model.save("cifar100_focused_model2.keras")
print("✅ Odaklanmış model kaydedildi.")

# Sonuç raporu
total_improvement = improvement_df['improvement'].mean() * 100
max_improvement = improvement_df['improvement_pct'].max()

print(f"\n🎉 SONUÇ RAPORU:")
print(f"📊 Genel Doğruluk: {test_accuracy:.4f}")
print(f"📈 Ortalama İyileşme: {total_improvement:.1f}%")
print(f"🚀 Maksimum İyileşme: {max_improvement:.1f}%")
print(f"🔧 İyileştirilen Sınıf Sayısı: {len(worst_15_classes)}")

# En çok iyileşen 5 sınıf
top_5_improved = improvement_df.head(5)
print(f"\n🏆 EN ÇOK İYİLEŞEN 5 SINIF:")
for _, row in top_5_improved.iterrows():
    print(f"  {row['class']}: {row['old_accuracy']:.3f} → {row['new_accuracy']:.3f} (+{row['improvement_pct']:.1f}%)")
