import pandas as pd
import re
from difflib import get_close_matches

# ==============================
# 1. تحميل البيانات
# ==============================
file_path = r"C:\Users\HP\OneDrive\الصور\سطح المكتب\car.ai2\extended_data.csv"
df = pd.read_csv(file_path)

# ==============================
# 2. تنظيف عمود brand
# ==============================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z]', '', text)  # حذف أي رموز أو مسافات
    return text

df['brand'] = df['brand'].apply(clean_text)

# ==============================
# 3. توحيد الأسماء (Normalization)
# ==============================
brand_mapping = {
    'mercedesbenz': 'mercedes',
    'benz': 'mercedes',
    'mercedes': 'mercedes',
    
    'bmw': 'bmw',
    
    'toyota': 'toyota',
    'toytoa': 'toyota',
    
    'honda': 'honda',
    
    'nissan': 'nissan',
    
    'hyundai': 'hyundai',
    'hundai': 'hyundai',
    
    'kia': 'kia',
    
    'chevrolet': 'chevrolet',
    'chevy': 'chevrolet',
    
    'ford': 'ford',
    
    'audi': 'audi'
}

df['brand'] = df['brand'].map(lambda x: brand_mapping.get(x, x))

# ==============================
# 4. تصحيح ذكي (Fuzzy Matching)
# ==============================
allowed_brands = [
    'audi', 'mercedes', 'bmw', 'toyota', 'honda',
    'nissan', 'hyundai', 'kia', 'chevrolet', 'ford'
]

def fix_brand(name):
    match = get_close_matches(name, allowed_brands, n=1, cutoff=0.8)
    return match[0] if match else None  # لو مش قريب → نحذفه

df['brand'] = df['brand'].apply(fix_brand)

# ==============================
# 5. حذف القيم غير المطلوبة
# ==============================
df_clean = df.dropna(subset=['brand'])

# ==============================
# 6. حفظ البيانات النظيفة
# ==============================
output_path = r"C:\Users\HP\OneDrive\الصور\سطح المكتب\car.ai2\cleaned_extended_data.csv"
df_clean.to_csv(output_path, index=False)

# ==============================
# 7. تقرير سريع
# ==============================
print("✅ تم تنظيف البيانات بنجاح")
print("عدد الصفوف قبل:", len(df))
print("عدد الصفوف بعد:", len(df_clean))

print("\n📊 عدد السيارات لكل شركة:")
print(df_clean['brand'].value_counts())