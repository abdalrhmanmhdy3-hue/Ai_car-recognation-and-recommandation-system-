import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# ==============================
# المسارات
# ==============================
train_path = r"C:\Users\HP\OneDrive\الصور\سطح المكتب\car.ai2\clean_dataset\train"
test_path  = r"C:\Users\HP\OneDrive\الصور\سطح المكتب\car.ai2\clean_dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# Data Augmentation
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ==============================
# تحميل VGG16
# ==============================
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# ==============================
# بناء الموديل
# ==============================
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.BatchNormalization(),

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(256, activation='relu'),

    layers.Dense(train_data.num_classes, activation='softmax')
])

# ==============================
# Compile
# ==============================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==============================
# التدريب
# ==============================
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=15
)

# ==============================
# حفظ الموديل
# ==============================
model.save("car_model_vgg166.h5")

print("✅ تم التدريب باستخدام VGG16")