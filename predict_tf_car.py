import argparse
from pathlib import Path

from core.tf_car_system import CarRecognitionRecommendationSystem
from project_config import MODEL_PATH, SPECS_CSV, TEST_DIR, TRAIN_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Predict car class and recommend cross-region cars.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--specs-csv", type=Path, default=SPECS_CSV)
    return parser.parse_args()


def main():
    args = parse_args()
    system = CarRecognitionRecommendationSystem(
        model_path=args.model_path,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        specs_csv_path=args.specs_csv,
    )
    payload = system.run(args.image_path)
    print(system.to_json(payload))


if __name__ == "__main__":
    main()
