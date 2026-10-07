import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


REGION_BY_BRAND = {
    "audi": "german",
    "mercedes": "german",
    "bmw": "german",
    "toyota": "japanese",
    "honda": "japanese",
    "nissan": "japanese",
    "hyundai": "korean",
    "hundai": "korean",
    "kia": "korean",
    "ford": "american",
    "chevrolet": "american",
    "chevrloet": "american",
}

BRAND_ALIASES = {
    "benz": "mercedes",
    "mercedes-benz": "mercedes",
    "mercedes benz": "mercedes",
    "hundai": "hyundai",
    "hunday": "hyundai",
    "chevrloet": "chevrolet",
    "chevy": "chevrolet",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("-", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_brand(value: str) -> str:
    normalized = normalize_text(value)
    return BRAND_ALIASES.get(normalized, normalized)


def normalize_model(value: str) -> str:
    normalized = normalize_text(value)
    for token in ["base", "sedan", "coupe", "roadster", "convertible", "wagon", "hatchback"]:
        normalized = normalized.replace(token, " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def parse_class_label(label: str) -> tuple[str, str]:
    parts = label.split("_", 1)
    brand = normalize_brand(parts[0])
    model = normalize_model(parts[1] if len(parts) > 1 else "")
    return brand, model


def similarity_score(a: str, b: str) -> float:
    a_norm = normalize_model(a)
    b_norm = normalize_model(b)
    if not a_norm or not b_norm:
        return 0.0

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    token_score = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
    sequence_score = SequenceMatcher(None, a_norm, b_norm).ratio()
    return 0.55 * token_score + 0.45 * sequence_score


def safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def find_first_image(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            return file
    return None


@dataclass
class PredictionPayload:
    predicted_label: str
    predicted_brand: str
    predicted_model: str
    predicted_region: str
    confidence: float
    top_predictions: List[dict]
    reference_car: dict
    recommendations: List[dict]


class CarRecognitionRecommendationSystem:
    def __init__(
        self,
        model_path: str | Path,
        train_dir: str | Path,
        test_dir: str | Path,
        specs_csv_path: str | Path,
        image_size: tuple[int, int] = (224, 224),
    ):
        self.model_path = Path(model_path)
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.specs_csv_path = Path(specs_csv_path)
        self.image_size = image_size

        self.class_names = self._load_class_names()
        self.model = tf.keras.models.load_model(self.model_path)
        self.specs_df = self._load_specs()

    def _load_class_names(self) -> List[str]:
        folders = [folder.name for folder in self.train_dir.iterdir() if folder.is_dir()]
        return sorted(folders)

    def _load_specs(self) -> pd.DataFrame:
        df = pd.read_csv(self.specs_csv_path)
        df.columns = [column.strip() for column in df.columns]
        rename_map = {
            "premium_v": "premium_version",
            "miles_per": "miles_per_gallon",
        }
        df = df.rename(columns=rename_map)

        df["brand_norm"] = df["brand"].map(normalize_brand)
        df["model_norm"] = df["model"].map(normalize_model)
        df["type_norm"] = df["type"].map(normalize_text)
        df["region"] = df["brand_norm"].map(lambda brand: REGION_BY_BRAND.get(brand, "unknown"))
        df["msrp"] = pd.to_numeric(df["msrp"], errors="coerce")
        df["miles_per_gallon"] = pd.to_numeric(df["miles_per_gallon"], errors="coerce")
        df["premium_version"] = pd.to_numeric(df["premium_version"], errors="coerce").fillna(0).astype(int)
        df["collection_car"] = pd.to_numeric(df["collection_car"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["msrp"]).reset_index(drop=True)
        return df

    def preprocess_image(self, image_source) -> np.ndarray:
        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source).convert("RGB")
        else:
            image = Image.open(image_source).convert("RGB")

        image = image.resize(self.image_size)
        image_array = np.asarray(image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, image_source, top_k: int = 5) -> List[dict]:
        image_array = self.preprocess_image(image_source)
        probabilities = self.model.predict(image_array, verbose=0)[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            predictions.append(
                {
                    "label": self.class_names[idx],
                    "confidence": float(probabilities[idx]),
                }
            )
        return predictions

    def _find_reference_car(self, predicted_label: str) -> dict:
        predicted_brand, predicted_model = parse_class_label(predicted_label)
        brand_rows = self.specs_df[self.specs_df["brand_norm"] == predicted_brand].copy()

        if brand_rows.empty:
            return {
                "brand": predicted_brand,
                "model": predicted_model,
                "region": REGION_BY_BRAND.get(predicted_brand, "unknown"),
                "msrp": None,
            }

        brand_rows["model_score"] = brand_rows["model_norm"].map(lambda value: similarity_score(predicted_model, value))
        brand_rows = brand_rows.sort_values(["model_score", "msrp"], ascending=[False, True])
        best = brand_rows.iloc[0]
        return {
            "model_year": int(best["model_year"]) if not pd.isna(best["model_year"]) else None,
            "brand": str(best["brand"]),
            "model": str(best["model"]),
            "type": str(best["type"]),
            "miles_per_gallon": safe_float(best["miles_per_gallon"], None),
            "premium_version": int(best["premium_version"]),
            "msrp": safe_float(best["msrp"], None),
            "collection_car": int(best["collection_car"]),
            "region": str(best["region"]),
            "class_label": predicted_label,
            "image_path": self.find_image_for_car(str(best["brand"]), str(best["model"])),
        }

    def find_image_for_car(self, brand: str, model: str) -> Optional[str]:
        brand_norm = normalize_brand(brand)
        model_norm = normalize_model(model)

        candidate_folders: List[Path] = []
        for root_dir in [self.train_dir, self.test_dir]:
            if root_dir.exists():
                candidate_folders.extend([folder for folder in root_dir.iterdir() if folder.is_dir()])

        best_folder = None
        best_score = 0.0
        for folder in candidate_folders:
            if "_" not in folder.name:
                continue
            folder_brand, folder_model = parse_class_label(folder.name)
            if folder_brand != brand_norm:
                continue
            score = similarity_score(model_norm, folder_model)
            if score > best_score:
                best_score = score
                best_folder = folder

        if best_folder is None or best_score < 0.20:
            return None

        image_path = find_first_image(best_folder)
        return str(image_path) if image_path else None

    def _candidate_score(self, reference_car: dict, candidate_row: pd.Series) -> float:
        reference_price = reference_car.get("msrp")
        candidate_price = safe_float(candidate_row["msrp"], 0.0)
        if not reference_price:
            return float("inf")

        price_gap = abs(candidate_price - reference_price)
        type_bonus = 0.0
        if reference_car.get("type") and normalize_text(reference_car["type"]) == normalize_text(candidate_row["type"]):
            type_bonus = 0.08 * reference_price

        premium_bonus = 0.0
        if int(candidate_row["premium_version"]) == int(reference_car.get("premium_version", 0)):
            premium_bonus = 0.04 * reference_price

        mpg_gap = abs(safe_float(candidate_row["miles_per_gallon"], 0.0) - safe_float(reference_car.get("miles_per_gallon"), 0.0))
        mpg_penalty = mpg_gap * 120
        return price_gap - type_bonus - premium_bonus + mpg_penalty

    def recommend_cross_region(self, reference_car: dict) -> List[dict]:
        reference_region = reference_car.get("region", "unknown")
        reference_brand = normalize_brand(reference_car.get("brand", ""))

        candidate_df = self.specs_df.copy()
        candidate_df = candidate_df[candidate_df["region"] != reference_region]
        candidate_df = candidate_df[candidate_df["brand_norm"] != reference_brand]

        target_regions = [region for region in ["german", "japanese", "korean", "american"] if region != reference_region]
        recommendations = []

        for region in target_regions:
            region_df = candidate_df[candidate_df["region"] == region].copy()
            if region_df.empty:
                continue

            region_df["recommendation_score"] = region_df.apply(
                lambda row: self._candidate_score(reference_car, row),
                axis=1,
            )
            region_df = region_df.sort_values(["recommendation_score", "msrp"], ascending=[True, True])
            best = region_df.iloc[0]

            recommendations.append(
                {
                    "target_region": region,
                    "model_year": int(best["model_year"]) if not pd.isna(best["model_year"]) else None,
                    "brand": str(best["brand"]),
                    "model": str(best["model"]),
                    "type": str(best["type"]),
                    "miles_per_gallon": safe_float(best["miles_per_gallon"], None),
                    "premium_version": int(best["premium_version"]),
                    "msrp": safe_float(best["msrp"], None),
                    "collection_car": int(best["collection_car"]),
                    "price_difference": abs(safe_float(best["msrp"], 0.0) - safe_float(reference_car.get("msrp"), 0.0)),
                    "image_path": self.find_image_for_car(str(best["brand"]), str(best["model"])),
                }
            )

        return recommendations

    def run(self, image_source, top_k: int = 5) -> PredictionPayload:
        predictions = self.predict(image_source=image_source, top_k=top_k)
        top_prediction = predictions[0]
        predicted_label = top_prediction["label"]
        predicted_brand, predicted_model = parse_class_label(predicted_label)
        predicted_region = REGION_BY_BRAND.get(predicted_brand, "unknown")

        reference_car = self._find_reference_car(predicted_label)
        recommendations = self.recommend_cross_region(reference_car)

        return PredictionPayload(
            predicted_label=predicted_label,
            predicted_brand=predicted_brand,
            predicted_model=predicted_model,
            predicted_region=predicted_region,
            confidence=float(top_prediction["confidence"]),
            top_predictions=predictions,
            reference_car=reference_car,
            recommendations=recommendations,
        )

    @staticmethod
    def to_dict(payload: PredictionPayload) -> dict:
        return {
            "predicted_label": payload.predicted_label,
            "predicted_brand": payload.predicted_brand,
            "predicted_model": payload.predicted_model,
            "predicted_region": payload.predicted_region,
            "confidence": payload.confidence,
            "top_predictions": payload.top_predictions,
            "reference_car": payload.reference_car,
            "recommendations": payload.recommendations,
        }

    @staticmethod
    def to_json(payload: PredictionPayload) -> str:
        return json.dumps(CarRecognitionRecommendationSystem.to_dict(payload), ensure_ascii=False, indent=2)
