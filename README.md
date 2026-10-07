# Car.ai.project

مشروع موقع ويب للتعرف على السيارة من الصورة ثم اقتراح 3 سيارات من صناعات مختلفة وبأسعار قريبة.

## Structure

- `website.py`: تشغيل موقع الويب
- `predict_tf_car.py`: اختبار التوقع والتوصية من سطر الأوامر
- `project_config.py`: مسارات النموذج والبيانات
- `core/tf_car_system.py`: منطق التوقع والتوصية
- `templates/index.html`: صفحة الموقع
- `static/style.css`: التنسيق
- `requirements.txt`: المكتبات المطلوبة
- `docs/README_WEB.md`: شرح إضافي

## Run

```bash
pip install -r requirements.txt
python website.py
```

ثم افتح:

```text
http://127.0.0.1:5000
```

## Retrain Model

```bash
python train_model.py
```

سيتم حفظ النموذج الجديد داخل:

```text
Car.ai.project/artifacts/
```

ولتحديث الموقع لاستخدام النموذج الجديد:

```bash
python update_model_path.py
