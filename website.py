from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, render_template, request, send_file, url_for

from core.tf_car_system import CarRecognitionRecommendationSystem
from project_config import MODEL_PATH, SPECS_CSV, TEST_DIR, TRAIN_DIR


UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_system = None


def load_system():
    global _system
    if _system is None:
        _system = CarRecognitionRecommendationSystem(
            model_path=MODEL_PATH,
            train_dir=TRAIN_DIR,
            test_dir=TEST_DIR,
            specs_csv_path=SPECS_CSV,
        )
    return _system


def format_price(value):
    if value in (None, "", "-"):
        return "-"
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return str(value)


def format_percent(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def prepare_upload(file_storage) -> Path:
    extension = Path(file_storage.filename).suffix.lower() or ".jpg"
    file_name = f"{uuid4().hex}{extension}"
    target_path = UPLOAD_DIR / file_name
    file_storage.save(target_path)
    return target_path


def build_image_url(path: str | None) -> str | None:
    if not path:
        return None
    return url_for("serve_image", path=path)


@app.template_filter("price")
def price_filter(value):
    return format_price(value)


@app.template_filter("percent")
def percent_filter(value):
    return format_percent(value)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html", result=None, error=None)

    uploaded_file = request.files.get("car_image")
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("index.html", result=None, error="يرجى رفع صورة سيارة أولاً.")

    try:
        image_path = prepare_upload(uploaded_file)
        system = load_system()
        payload = system.run(image_path)
        result = system.to_dict(payload)
        result["uploaded_image_path"] = build_image_url(str(image_path))
        if result.get("reference_car"):
            result["reference_car"]["image_url"] = build_image_url(result["reference_car"].get("image_path"))
        for car in result.get("recommendations", []):
            car["image_url"] = build_image_url(car.get("image_path"))
        return render_template("index.html", result=result, error=None)
    except Exception as exc:
        return render_template("index.html", result=None, error=str(exc))


@app.route("/image")
def serve_image():
    raw_path = request.args.get("path", "").strip()
    if not raw_path:
        abort(404)

    image_path = Path(raw_path).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        abort(404)

    return send_file(image_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
