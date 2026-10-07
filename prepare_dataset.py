import os
import shutil
import random

# =====================================
# 📂 المسارات
# =====================================
dataset_path = r"C:\Users\HP\OneDrive\الصور\سطح المكتب\car.ai\dataset"
output_path  = r"C:\Users\HP\OneDrive\الصور\سطح المكتب\car.ai\clean_dataset"

train_path = os.path.join(output_path, "train")
test_path  = os.path.join(output_path, "test")

# =====================================
# ⚙️ الإعدادات
# =====================================
MIN_IMAGES = 150     # أقل عدد صور للكلاس
TRAIN_SPLIT = 0.8
MAX_IMAGES = 200     # (اختياري) حد أقصى لكل كلاس - خليه None لو مش عايز

# إنشاء الفولدرات
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# =====================================
# 🔄 المعالجة
# =====================================
valid_classes = 0
total_images = 0

for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_dir):
        continue

    images = [img for img in os.listdir(class_dir)
              if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # ❌ حذف الكلاسات الصغيرة
    if len(images) < MIN_IMAGES:
        continue

    # 🔀 ترتيب عشوائي
    random.shuffle(images)

    # ✨ تقليل العدد لو عايز توازن
    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    # ✂️ تقسيم
    split_index = int(len(images) * TRAIN_SPLIT)
    train_images = images[:split_index]
    test_images  = images[split_index:]

    # 📁 إنشاء فولدرات
    train_class_dir = os.path.join(train_path, class_name)
    test_class_dir  = os.path.join(test_path, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # 📥 نسخ الصور
    for img in train_images:
        shutil.copy2(os.path.join(class_dir, img),
                     os.path.join(train_class_dir, img))

    for img in test_images:
        shutil.copy2(os.path.join(class_dir, img),
                     os.path.join(test_class_dir, img))

    valid_classes += 1
    total_images += len(images)

# =====================================
# 📊 النتائج
# =====================================
print("✅ عدد الكلاسات بعد التنظيف:", valid_classes)
print("📸 إجمالي الصور المستخدمة:", total_images)
print("📁 Train/Test جاهزين في:", output_path)