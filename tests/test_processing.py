import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.processing import DataProcessor


class DataProcessorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.processor = DataProcessor(
            data_path=self.root / "orders.csv",
            rates_path=self.root / "rates.csv",
            output_path=self.root / "results",
            env_path=self.root / ".env",
            revenue_quantile=1,
            orders_quantile=1,
        )
        rows = [
            (1, "001", "2024-06-01", 100, 1, "rub", "концерты"),
            (2, "001", "2024-06-01", 100, 9, "rub", "концерты"),
            (3, "001", "2024-06-02", -5, 1, "rub", "концерты"),
            (4, "001", "2024-06-04", 0, 1, "rub", "концерты"),
            (5, "002", "2024-06-10", 1000, 2, "kzt", "спорт"),
            (6, "003", "2024-08-10", 50, 1, "rub", "концерты"),
            (7, "004", "2024-06-07", 20, 0, "rub", "концерты"),
        ]
        orders = pd.DataFrame(
            rows,
            columns=[
                "order_id",
                "user_id",
                "order_dt",
                "revenue",
                "tickets_count",
                "currency_code",
                "event_type_main",
            ],
        )
        orders["order_ts"] = orders.order_dt + " 12:00:00"
        orders["event_id"] = 10
        orders["device_type_canonical"] = " Mobile "
        orders["service_name"] = "Афиша"
        orders["region_name"] = "Регион"
        orders["city_name"] = "Город"
        orders.iloc[::-1].to_csv(self.processor.data_path, index=False)
        pd.DataFrame(
            {
                "data": orders.order_dt.unique(),
                "curs": 20,
                "nominal": 100,
            }
        ).to_csv(self.processor.rates_path, index=False)
        self.processor.load_data()
        self.processor.load_exchange_rates()

    def test_deduplication_uses_business_fields_and_keeps_first_id(self):
        clean = self.processor.preprocess_data()
        self.assertEqual(self.processor.audit["implicit_duplicates"], 1)
        self.assertEqual(self.processor.audit["duplicate_users"], 1)
        self.assertEqual(set(clean.order_id), {1, 4, 5, 6})
        self.assertEqual(clean.loc[clean.order_id.eq(1), "tickets_count"].item(), 1)

    def test_conversion_zero_revenue_and_intervals_after_cleaning(self):
        clean = self.processor.preprocess_data().set_index("order_id")
        self.assertAlmostEqual(clean.loc[5, "revenue_rub"], 200)
        self.assertEqual(clean.loc[4, "revenue_rub"], 0)
        self.assertEqual(clean.loc[4, "days_since_prev"], 3)
        self.assertTrue(pd.isna(clean.loc[1, "days_since_prev"]))
        self.assertEqual(self.processor.audit["negative_revenue"], 1)
        self.assertEqual(self.processor.audit["invalid_tickets"], 1)

    def test_profiles_preserve_identifiers_and_single_order_missing_interval(self):
        self.processor.preprocess_data()
        profiles = self.processor.create_user_profiles()
        self.assertEqual(profiles.loc["001", "total_orders"], 2)
        self.assertTrue(profiles.loc["001", "is_two"])
        self.assertTrue(pd.isna(profiles.loc["002", "avg_days_between_orders"]))
        self.assertEqual(profiles.loc["001", "order_group"], "2")

    def test_30_day_window_excludes_incomplete_observation(self):
        self.processor.preprocess_data()
        self.processor.create_user_profiles()
        retention = self.processor.calculate_retention()
        self.assertEqual(set(self.processor.profiles_30d.index), {"001", "002"})
        self.assertAlmostEqual(self.processor.profiles_30d.is_two.mean(), 0.5)
        row = retention["first_device"].loc["mobile"]
        self.assertEqual(row.users, 3)
        self.assertAlmostEqual(row.return_rate, 1 / 3)
        self.assertLess(row.ci_low, row.return_rate)
        self.assertGreater(row.ci_high, row.return_rate)

    def test_hypothesis_uses_observed_sport_and_concert_segments(self):
        self.processor.preprocess_data()
        self.processor.create_user_profiles()
        self.processor.calculate_retention()
        hypothesis = self.processor.check_hypotheses(1)["sport_vs_concert"]
        self.assertFalse(hypothesis["supported_descriptively"])
        self.assertEqual(hypothesis["sport_return_rate"], 0)
        self.assertEqual(hypothesis["concert_return_rate"], 0.5)

    def test_30_day_window_includes_day_30_and_excludes_day_31(self):
        original = self.processor.raw_data.copy()
        for elapsed_days, expected in [(30, True), (31, False)]:
            with self.subTest(elapsed_days=elapsed_days):
                self.processor.raw_data = original.copy()
                date = pd.Timestamp("2024-06-01") + pd.Timedelta(days=elapsed_days)
                selected = self.processor.raw_data.order_id.eq(4)
                self.processor.raw_data.loc[selected, "order_dt"] = str(date.date())
                self.processor.raw_data.loc[selected, "order_ts"] = (
                    str(date.date()) + " 12:00:00"
                )
                self.processor.preprocess_data()
                self.processor.create_user_profiles()
                self.processor.calculate_retention()
                self.assertEqual(
                    self.processor.profiles_30d.loc["001", "is_two"], expected
                )

    def test_missing_kzt_rate_is_rejected(self):
        rates = self.processor.exchange_rates
        self.processor.exchange_rates = rates.loc[
            rates.order_dt.ne(pd.Timestamp("2024-06-10"))
        ]
        with self.assertRaisesRegex(ValueError, "отсутствует дневной курс"):
            self.processor.preprocess_data()

    def test_duplicate_rate_date_is_rejected(self):
        rates = pd.read_csv(self.processor.rates_path)
        pd.concat([rates, rates.iloc[:1]]).to_csv(
            self.processor.rates_path, index=False
        )
        with self.assertRaisesRegex(ValueError, "уникальна"):
            self.processor.load_exchange_rates()

    def test_missing_data_without_database_settings_has_clear_error(self):
        self.processor.data_path.unlink()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "заполните параметры DB_"):
                self.processor.load_data()


if __name__ == "__main__":
    unittest.main()
