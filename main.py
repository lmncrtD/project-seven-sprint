import sys
from pathlib import Path

from src.processing import DataProcessor


def main():
    root = Path(__file__).resolve().parent
    processor = DataProcessor(
        data_path=root / "data/orders.csv",
        rates_path=root / "data/final_tickets_tenge_df.csv",
        output_path=root / "results",
        env_path=root / ".env",
    )
    try:
        processor.load_data()
        processor.load_exchange_rates()
        processor.preprocess_data()
        processor.create_user_profiles()
        processor.calculate_retention()
        processor.check_hypotheses()
        processor.analyze_correlations()
        processor.save_results()
        print(processor.build_report())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
