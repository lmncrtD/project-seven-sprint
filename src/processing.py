import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.figure import Figure
from phik.phik import phik_matrix
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError


class DataProcessor:
    def __init__(
        self,
        data_path="data/orders.csv",
        rates_path="data/final_tickets_tenge_df.csv",
        output_path="results",
        env_path=".env",
        revenue_quantile=0.99,
        orders_quantile=0.99,
    ):
        if not 0 < revenue_quantile <= 1 or not 0 < orders_quantile <= 1:
            raise ValueError("Квантили должны находиться в интервале (0, 1].")
        load_dotenv(dotenv_path=env_path)
        self.data_path = Path(data_path)
        self.rates_path = Path(rates_path)
        self.output_path = Path(output_path)
        self.revenue_quantile = revenue_quantile
        self.orders_quantile = orders_quantile
        self.raw_data = None
        self.exchange_rates = None
        self.clean_data = None
        self.all_profiles = None
        self.profiles = None
        self.profiles_30d = None
        self.retention = {}
        self.hypotheses = {}
        self.correlations = {}
        self.audit = {}

    def load_data(self):
        if not self.data_path.exists():
            self.export_orders_from_database()
        self.raw_data = pd.read_csv(self.data_path, dtype={"user_id": "string"})
        if self.raw_data.empty:
            raise ValueError("Выгрузка заказов пуста.")
        return self.raw_data

    def export_orders_from_database(self):
        names = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
        configuration = {name: os.getenv(name) for name in names}
        if not all(configuration.values()):
            raise ValueError(
                "Нет data/orders.csv. Поместите SQL-выгрузку в data/ "
                "или заполните параметры DB_* в .env."
            )
        try:
            port = int(configuration["DB_PORT"])
        except ValueError:
            raise ValueError("DB_PORT должен быть целым числом.") from None
        url = URL.create(
            "postgresql+psycopg2",
            username=configuration["DB_USER"],
            password=configuration["DB_PASSWORD"],
            host=configuration["DB_HOST"],
            port=port,
            database=configuration["DB_NAME"],
        )
        engine = create_engine(
            url,
            connect_args={
                "sslmode": os.getenv("DB_SSLMODE", "require"),
                "connect_timeout": 20,
            },
        )
        query_path = Path(__file__).with_name("query.sql")
        try:
            with engine.connect().execution_options(
                postgresql_readonly=True
            ) as connection:
                orders = pd.read_sql_query(
                    query_path.read_text(encoding="utf-8"), connection
                )
        except SQLAlchemyError:
            raise RuntimeError(
                "Не удалось выгрузить заказы. Проверьте доступ к БД и .env."
            ) from None
        finally:
            engine.dispose()
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        orders.to_csv(self.data_path, index=False)
        return orders

    def load_exchange_rates(self):
        self.exchange_rates = pd.read_csv(self.rates_path)
        required = {"data", "curs", "nominal"}
        if not required.issubset(self.exchange_rates.columns):
            raise ValueError("В файле курса нужны столбцы data, curs и nominal.")
        rates = self.exchange_rates.copy()
        rates["order_dt"] = pd.to_datetime(rates["data"], errors="raise").dt.normalize()
        for column in ["curs", "nominal"]:
            rates[column] = pd.to_numeric(rates[column], errors="raise")
            if not np.isfinite(rates[column]).all() or not rates[column].gt(0).all():
                raise ValueError("Курс и номинал должны быть положительными числами.")
        if rates.order_dt.isna().any() or not rates.order_dt.is_unique:
            raise ValueError("Дата курса должна быть заполнена и уникальна.")
        rates["rub_per_kzt"] = rates.curs / rates.nominal
        self.exchange_rates = rates
        return rates

    def preprocess_data(self):
        if self.raw_data is None or self.exchange_rates is None:
            raise ValueError("Сначала загрузите заказы и курсы валют.")
        required = {
            "user_id",
            "order_id",
            "order_dt",
            "order_ts",
            "currency_code",
            "revenue",
            "tickets_count",
            "event_id",
            "device_type_canonical",
            "event_type_main",
            "service_name",
            "region_name",
            "city_name",
        }
        missing = required.difference(self.raw_data.columns)
        if missing:
            raise ValueError("Нет обязательных столбцов: " + ", ".join(sorted(missing)))
        data = self.raw_data.copy()
        if data[list(required)].isna().any().any():
            raise ValueError("В обязательных столбцах заказов есть пропуски.")
        if not data.order_id.is_unique:
            raise ValueError("Идентификаторы order_id должны быть уникальны.")
        data["user_id"] = data.user_id.astype("string")
        for column in ["order_dt", "order_ts"]:
            data[column] = pd.to_datetime(data[column], errors="raise")
        data["order_dt"] = data.order_dt.dt.normalize()
        if not data.order_dt.eq(data.order_ts.dt.normalize()).all():
            raise ValueError("Дата заказа не совпадает с датой его времени.")
        categorical = [
            "currency_code",
            "device_type_canonical",
            "event_type_main",
            "service_name",
            "region_name",
            "city_name",
        ]
        for column in categorical:
            data[column] = (
                data[column]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
        for column in ["currency_code", "device_type_canonical"]:
            data[column] = data[column].str.lower()
        data = data.loc[
            data.device_type_canonical.isin(["mobile", "desktop"])
            & data.event_type_main.ne("фильм")
        ].copy()
        if data.empty:
            raise ValueError("Нет заказов выбранных устройств и типов мероприятий.")
        if not set(data.currency_code.unique()) <= {"rub", "kzt"}:
            raise ValueError("В заказах есть неподдерживаемая валюта.")
        for column in ["order_id", "event_id", "revenue", "tickets_count"]:
            data[column] = pd.to_numeric(data[column], errors="raise")
            if not np.isfinite(data[column]).all():
                raise ValueError("Неконечные значения в столбце " + column)
        for column in ["order_id", "event_id", "tickets_count"]:
            if not data[column].mod(1).eq(0).all():
                raise ValueError("Столбец должен содержать целые числа: " + column)
        data = data.sort_values(["user_id", "order_ts", "order_id"], kind="stable")
        self.raw_data = data.copy()
        duplicate_fields = [
            "user_id",
            "order_ts",
            "device_type_canonical",
            "service_name",
            "event_id",
            "revenue",
        ]
        duplicates = data.duplicated(duplicate_fields, keep="first")
        counts_before = data.groupby("user_id").order_id.nunique()
        duplicate_count = int(duplicates.sum())
        duplicate_users = int(data.loc[duplicates, "user_id"].nunique())
        data = data.loc[~duplicates].copy()
        counts_after = data.groupby("user_id").order_id.nunique()
        repeat_changes = int(counts_before.ge(2).ne(counts_after.ge(2)).sum())
        data = data.merge(
            self.exchange_rates[["order_dt", "rub_per_kzt"]],
            on="order_dt",
            how="left",
            validate="many_to_one",
        )
        kzt = data.currency_code.eq("kzt")
        if data.loc[kzt, "rub_per_kzt"].isna().any():
            raise ValueError("Для части заказов в тенге отсутствует дневной курс.")
        data["revenue_rub"] = np.where(
            kzt, data.revenue * data.rub_per_kzt, data.revenue
        )
        nonnegative = data.loc[data.revenue_rub.ge(0), "revenue_rub"]
        if nonnegative.empty:
            raise ValueError("Нет заказов с неотрицательной выручкой.")
        revenue_cap = float(nonnegative.quantile(self.revenue_quantile))
        negative = data.revenue_rub.lt(0)
        high_revenue = data.revenue_rub.gt(revenue_cap)
        bad_tickets = data.tickets_count.le(0)
        clean = data.loc[~negative & ~high_revenue & ~bad_tickets].copy()
        if clean.empty:
            raise ValueError("После очистки не осталось заказов.")
        clean = clean.sort_values(["user_id", "order_ts", "order_id"], kind="stable")
        clean["days_since_prev"] = (
            clean.groupby("user_id").order_dt.diff().dt.days.astype("Int16")
        )
        for column in categorical:
            clean[column] = clean[column].astype("category")
        self.audit = {
            "raw_orders": len(self.raw_data),
            "raw_users": int(self.raw_data.user_id.nunique()),
            "implicit_duplicates": duplicate_count,
            "duplicate_users": duplicate_users,
            "dedup_repeat_changes": repeat_changes,
            "negative_revenue": int(negative.sum()),
            "invalid_tickets": int(bad_tickets.sum()),
            "high_revenue": int(high_revenue.sum()),
            "revenue_cap_rub": revenue_cap,
            "clean_orders": len(clean),
            "clean_users": int(clean.user_id.nunique()),
            "removed_fraction": 1 - len(clean) / len(self.raw_data),
        }
        self.clean_data = clean
        return clean

    def create_user_profiles(self):
        if self.clean_data is None:
            raise ValueError("Сначала выполните предобработку заказов.")
        profiles = self.clean_data.groupby("user_id", sort=False).agg(
            first_order_dt=("order_dt", "first"),
            last_order_dt=("order_dt", "last"),
            first_order_ts=("order_ts", "first"),
            first_order_id=("order_id", "first"),
            first_device=("device_type_canonical", "first"),
            first_region=("region_name", "first"),
            first_event=("event_type_main", "first"),
            first_revenue_rub=("revenue_rub", "first"),
            first_tickets=("tickets_count", "first"),
            total_orders=("order_id", "nunique"),
            avg_revenue_rub=("revenue_rub", "mean"),
            avg_tickets=("tickets_count", "mean"),
            total_revenue_rub=("revenue_rub", "sum"),
            total_tickets=("tickets_count", "sum"),
            avg_days_between_orders=("days_since_prev", "mean"),
        )
        profiles["avg_days_between_orders"] = profiles.avg_days_between_orders.astype(
            float
        )
        profiles["avg_revenue_per_ticket"] = (
            profiles.total_revenue_rub / profiles.total_tickets
        )
        profiles["is_two"] = profiles.total_orders.ge(2)
        profiles["first_weekday"] = profiles.first_order_ts.dt.dayofweek
        period_end = self.clean_data.order_dt.max()
        profiles["observation_days"] = (period_end - profiles.first_order_dt).dt.days
        self.all_profiles = profiles
        orders_cap = float(profiles.total_orders.quantile(self.orders_quantile))
        self.profiles = profiles.loc[profiles.total_orders.le(orders_cap)].copy()
        self.profiles["order_group"] = pd.cut(
            self.profiles.total_orders,
            bins=[0, 1, 2, 4, np.inf],
            labels=["1", "2", "3-4", "5+"],
        ).astype(str)
        removed = profiles.loc[profiles.total_orders.gt(orders_cap)]
        raw_counts = self.raw_data.groupby("user_id").order_id.nunique()
        raw_first = self.raw_data.groupby("user_id", sort=False).order_id.first()
        self.audit.update(
            {
                "orders_cap": orders_cap,
                "removed_active_users": len(removed),
                "removed_active_order_fraction": float(
                    removed.total_orders.sum() / profiles.total_orders.sum()
                ),
                "lost_users": len(raw_counts.index.difference(profiles.index)),
                "changed_first_orders": int(
                    raw_first.reindex(profiles.index).ne(profiles.first_order_id).sum()
                ),
                "changed_repeat_labels": int(
                    raw_counts.reindex(profiles.index).ge(2).ne(profiles.is_two).sum()
                ),
            }
        )
        return self.profiles

    @staticmethod
    def _segment_summary(profiles, feature):
        summary = profiles.groupby(feature, observed=True).agg(
            users=("is_two", "size"), returned=("is_two", "sum")
        )
        summary["share"] = summary.users / len(profiles)
        summary["return_rate"] = summary.returned / summary.users
        z = 1.959964
        n = summary.users
        p = summary.return_rate
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
        summary["ci_low"] = center - half
        summary["ci_high"] = center + half
        summary["difference_pp"] = (p - profiles.is_two.mean()) * 100
        return summary.sort_values("users", ascending=False)

    def calculate_retention(self):
        if self.profiles is None:
            raise ValueError("Сначала создайте профили пользователей.")
        features = ["first_event", "first_device", "first_region", "first_weekday"]
        self.retention = {
            feature: self._segment_summary(self.profiles, feature)
            for feature in features
        }
        for feature in features:
            self.retention[feature + "_all_users"] = self._segment_summary(
                self.all_profiles, feature
            )
        eligible = self.profiles.loc[self.profiles.observation_days.ge(30)].copy()
        if eligible.empty:
            raise ValueError("Нет пользователей с полным 30-дневным наблюдением.")
        window = self.clean_data.loc[
            self.clean_data.user_id.isin(eligible.index)
        ].merge(
            eligible[["first_order_dt"]],
            left_on="user_id",
            right_index=True,
            validate="many_to_one",
        )
        elapsed = (window.order_dt - window.first_order_dt).dt.days
        counts = (
            window.loc[elapsed.between(0, 30)].groupby("user_id").order_id.nunique()
        )
        eligible["is_two"] = counts.reindex(eligible.index, fill_value=0).ge(2)
        self.profiles_30d = eligible
        for feature in features:
            self.retention[feature + "_30d"] = self._segment_summary(eligible, feature)
        return self.retention

    def check_hypotheses(self, minimum_segment_users=100):
        if not self.retention:
            raise ValueError("Сначала рассчитайте возвраты по сегментам.")
        events = self.retention["first_event"]
        comparison = events.reindex(["спорт", "концерты"])
        sport_rate = comparison.loc["спорт", "return_rate"]
        concert_rate = comparison.loc["концерты", "return_rate"]
        available = pd.notna(sport_rate) and pd.notna(concert_rate)
        regions = self.retention["first_region"].loc[
            lambda table: table.users.ge(minimum_segment_users)
        ]
        major_events = events.loc[events.users.ge(minimum_segment_users)]
        metrics = ["avg_revenue_rub", "avg_tickets", "avg_revenue_per_ticket"]
        purchase_summary = self.profiles.groupby("is_two")[metrics].agg(
            ["count", "mean", "median"]
        )
        revenue_quantiles = (
            self.profiles.groupby("is_two")
            .avg_revenue_rub.quantile([0.25, 0.5, 0.75])
            .unstack()
        )
        self.hypotheses = {
            "sport_vs_concert": {
                "sport_return_rate": float(sport_rate) if available else None,
                "concert_return_rate": float(concert_rate) if available else None,
                "difference_pp": (
                    float((sport_rate - concert_rate) * 100) if available else None
                ),
                "supported_descriptively": (
                    bool(sport_rate > concert_rate) if available else None
                ),
            },
            "minimum_segment_users": minimum_segment_users,
            "best_event": (
                str(major_events.return_rate.idxmax())
                if not major_events.empty
                else None
            ),
            "best_region": (
                str(regions.return_rate.idxmax()) if not regions.empty else None
            ),
            "major_regions": regions,
            "region_size_comparison": {
                "regions": len(regions),
                "min_return_rate": (
                    float(regions.return_rate.min()) if not regions.empty else None
                ),
                "max_return_rate": (
                    float(regions.return_rate.max()) if not regions.empty else None
                ),
                "largest_region": (
                    str(regions.users.idxmax()) if not regions.empty else None
                ),
            },
            "purchase_summary": purchase_summary,
            "revenue_quantiles": revenue_quantiles,
        }
        return self.hypotheses

    def analyze_correlations(self):
        if self.profiles is None:
            raise ValueError("Сначала создайте профили пользователей.")
        base_columns = [
            "total_orders",
            "first_device",
            "first_region",
            "first_event",
            "first_weekday",
            "first_revenue_rub",
            "first_tickets",
            "avg_revenue_rub",
            "avg_tickets",
            "observation_days",
        ]
        interval_columns = [
            "total_orders",
            "first_revenue_rub",
            "first_tickets",
            "avg_revenue_rub",
            "avg_tickets",
            "observation_days",
        ]
        group_columns = ["order_group"] + [
            column for column in base_columns if column != "total_orders"
        ]
        repeat_columns = [
            "total_orders",
            "avg_days_between_orders",
            "avg_revenue_rub",
            "avg_tickets",
        ]
        self.correlations = {
            "numeric": phik_matrix(
                self.profiles[base_columns],
                interval_cols=interval_columns,
                njobs=1,
                verbose=False,
            ),
            "groups": phik_matrix(
                self.profiles[group_columns],
                interval_cols=[
                    column for column in interval_columns if column != "total_orders"
                ],
                njobs=1,
                verbose=False,
            ),
            "repeat": phik_matrix(
                self.profiles.loc[self.profiles.is_two, repeat_columns],
                interval_cols=repeat_columns,
                njobs=1,
                verbose=False,
            ),
        }
        return self.correlations

    def build_report(self):
        if not self.hypotheses or not self.correlations:
            raise ValueError("Перед отчетом выполните анализ гипотез и корреляций.")
        audit = self.audit
        sport = self.hypotheses["sport_vs_concert"]
        medians = self.hypotheses["revenue_quantiles"][0.5]
        regions = self.hypotheses["region_size_comparison"]
        weekdays = self.retention["first_weekday"].return_rate
        ranking = (
            self.correlations["groups"]
            .order_group.drop("order_group")
            .dropna()
            .sort_values(ascending=False)
            .head(3)
        )
        period_start = self.raw_data.order_dt.min().date()
        period_end = self.raw_data.order_dt.max().date()
        lines = [
            "# Анализ лояльности пользователей Яндекс Афиши",
            "",
            f"Период: {period_start} - {period_end}.",
            f"Выгрузка: {audit['raw_orders']:,} заказов и "
            f"{audit['raw_users']:,} пользователей.",
            f"Удалено неявных дубликатов: {audit['implicit_duplicates']}. "
            f"После очистки: {audit['clean_orders']:,} заказов и "
            f"{audit['clean_users']:,} пользователей. "
            f"Потери заказов: {audit['removed_fraction']:.2%}.",
            f"Основная выборка: {len(self.profiles):,} пользователей. "
            f"Исключено {audit['removed_active_users']} профилей "
            f"с числом заказов выше P99 ({audit['orders_cap']:.0f}).",
            "",
            "## Результаты",
            "",
            f"- Доля пользователей с двумя и более заказами: "
            f"{self.profiles.is_two.mean():.2%}. "
            f"До ограничения активности: {self.all_profiles.is_two.mean():.2%}.",
            f"- Возврат в течение 30 календарных дней: "
            f"{self.profiles_30d.is_two.mean():.2%} "
            f"среди {len(self.profiles_30d):,} пользователей "
            "с полным окном наблюдения.",
        ]
        for device, row in self.retention["first_device"].iterrows():
            lines.append(
                f"- Первое устройство {device}: возврат {row.return_rate:.2%}, "
                f"размер сегмента {int(row.users):,}."
            )
        if sport["sport_return_rate"] is not None:
            direction = "выше" if sport["supported_descriptively"] else "ниже"
            lines.append(
                f"- Спорт: {sport['sport_return_rate']:.2%}; "
                f"концерты: {sport['concert_return_rate']:.2%}. "
                f"Наблюдаемый возврат после спорта {direction}, чем после концертов."
            )
        lines.extend(
            [
                f"- Среди сегментов со 100 и более пользователями "
                f"максимальный возврат по типу первого события: "
                f"{self.hypotheses['best_event']}; по региону: "
                f"{self.hypotheses['best_region']}.",
            ]
        )
        if False in medians.index and True in medians.index:
            lines.append(
                f"- Медиана средней выручки с заказа: {medians.loc[False]:.1f} руб. "
                f"у разовых и {medians.loc[True]:.1f} руб. у повторных покупателей."
            )
        if regions["regions"]:
            lines.append(
                f"- Среди {regions['regions']} регионов со 100 и более пользователями "
                f"возврат составляет {regions['min_return_rate']:.2%}"
                f"-{regions['max_return_rate']:.2%}. Размер региона сам по себе "
                "не объясняет различия возврата."
            )
        lines.extend(
            [
                f"- Возврат по дням первой покупки: {weekdays.min():.2%}"
                f"-{weekdays.max():.2%}. Различия могут зависеть от состава аудитории.",
                "- Наибольшая связь с группой числа заказов по PhiK: "
                + ", ".join(f"{name} ({value:.3f})" for name, value in ranking.items())
                + ".",
                "",
                "## Рекомендации",
                "",
                "- Начать тесты удержания с концертов и mobile: это крупные сегменты "
                "для персональных подборок и напоминаний о новых мероприятиях.",
                "- Проверить сегменты с высоким возвратом отдельными экспериментами. "
                "Учитывать размер аудитории и 95% интервалы долей.",
                "- Ультраактивные аккаунты изучить отдельно, прежде чем включать "
                "их в массовые маркетинговые кампании.",
                "- Для прогнозирования использовать признаки, известные на дату "
                "прогноза, и одинаковые окна наблюдения.",
                "",
                "## Ограничения",
                "",
                "Возврат определен как наличие двух и более заказов в доступном окне. "
                "Это не классический когортный Retention. Первое событие означает "
                "первый наблюдаемый заказ, а не первую покупку за всю жизнь клиента. "
                "Основное окно неодинаково для разных пользователей; проверка за "
                "30 дней частично устраняет это различие. Интервалы Уилсона и "
                "описательные сравнения не доказывают причинность. Средние покупки "
                "и длительность наблюдения могут зависеть от самого числа заказов. "
                "PhiK показывает связь без направления. Удаление выбросов ограничивает "
                "применимость выводов к самым активным пользователям.",
            ]
        )
        return "\n".join(lines) + "\n"

    def save_results(self):
        if not self.correlations or not self.hypotheses:
            raise ValueError("Перед сохранением выполните весь анализ.")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.all_profiles.to_csv(self.output_path / "user_profiles_all.csv")
        self.profiles.to_csv(self.output_path / "user_profiles.csv")
        for name, table in self.retention.items():
            table.to_csv(self.output_path / f"retention_{name}.csv")
        for name, table in self.correlations.items():
            table.to_csv(self.output_path / f"phik_{name}.csv")
        for name in ["major_regions", "purchase_summary", "revenue_quantiles"]:
            self.hypotheses[name].to_csv(self.output_path / f"{name}.csv")
        metrics = {
            **self.audit,
            "main_users": len(self.profiles),
            "main_orders": int(self.profiles.total_orders.sum()),
            "repeat_rate": float(self.profiles.is_two.mean()),
            "repeat_rate_all": float(self.all_profiles.is_two.mean()),
            "eligible_30d_users": len(self.profiles_30d),
            "repeat_rate_30d": float(self.profiles_30d.is_two.mean()),
            "sport_vs_concert": self.hypotheses["sport_vs_concert"],
            "best_event": self.hypotheses["best_event"],
            "best_region": self.hypotheses["best_region"],
            "region_size_comparison": self.hypotheses["region_size_comparison"],
        }
        (self.output_path / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (self.output_path / "report.md").write_text(
            self.build_report(), encoding="utf-8"
        )
        self._save_figures()

    def _save_figure(self, figure, name):
        figure.tight_layout()
        figure.savefig(self.output_path / f"{name}.png", dpi=150, bbox_inches="tight")
        figure.clear()

    def _plot_segments(self, table, title, name):
        segment = table.head(10)
        positions = np.arange(len(segment))
        figure = Figure(figsize=(14, max(4, len(segment) * 0.45)))
        axes = figure.subplots(1, 2)
        axes[0].barh(positions, segment.users, color="#4477aa")
        axes[0].set(title=title, xlabel="Пользователи")
        axes[1].errorbar(
            segment.return_rate * 100,
            positions,
            xerr=np.maximum(
                0,
                np.vstack(
                    [
                        (segment.return_rate - segment.ci_low) * 100,
                        (segment.ci_high - segment.return_rate) * 100,
                    ]
                ),
            ),
            fmt="o",
            color="#228833",
            capsize=3,
        )
        axes[1].axvline(
            self.profiles.is_two.mean() * 100,
            color="#cc6677",
            linestyle="--",
            label="Основная выборка",
        )
        axes[1].set(xlabel="Доля 2+ заказов, % (95% интервал)", title="Возврат")
        axes[1].legend()
        for axis in axes:
            labels = segment.index.astype(str).str.replace("ё", "е")
            axis.set_yticks(positions, labels)
            axis.invert_yaxis()
        self._save_figure(figure, name)

    def _save_figures(self):
        sns.set_theme(style="whitegrid", font="DejaVu Sans")
        figure = Figure(figsize=(12, 8))
        axes = figure.subplots(2, 2).ravel()
        for axis, column, title in zip(
            axes,
            ["revenue_rub", "tickets_count", "days_since_prev", "rub_per_kzt"],
            [
                "Выручка заказа, руб.",
                "Билеты в заказе",
                "Интервал, дни",
                "Рублей за тенге",
            ],
        ):
            sns.histplot(self.clean_data[column].dropna(), bins=40, ax=axis)
            axis.set_title(title)
        self._save_figure(figure, "order_distributions")

        figure = Figure(figsize=(12, 4))
        axes = figure.subplots(1, 2)
        sns.histplot(self.all_profiles.total_orders, bins=70, ax=axes[0])
        axes[0].set(xscale="log", title="Число заказов до ограничения P99")
        sns.histplot(
            self.all_profiles.avg_days_between_orders.dropna(), bins=40, ax=axes[1]
        )
        axes[1].set_title("Средний интервал у повторных покупателей, дни")
        self._save_figure(figure, "profile_distributions")

        for feature, title in [
            ("first_event", "Тип первого события"),
            ("first_device", "Первое устройство"),
            ("first_region", "Первый регион"),
        ]:
            self._plot_segments(self.retention[feature], title, feature)

        figure = Figure(figsize=(8, 5))
        axis = figure.subplots()
        regions = self.hypotheses["major_regions"]
        axis.scatter(regions.users, regions.return_rate * 100, alpha=0.7)
        axis.axhline(self.profiles.is_two.mean() * 100, color="#cc6677", linestyle="--")
        axis.set(
            xscale="log",
            xlabel="Пользователи региона (log)",
            ylabel="Доля 2+ заказов, %",
            title="Размер региона и наблюдаемый возврат",
        )
        self._save_figure(figure, "region_size_and_return")

        figure = Figure(figsize=(16, 4))
        axes = figure.subplots(1, 3)
        for axis, column, title in zip(
            axes,
            ["avg_revenue_rub", "avg_tickets", "avg_revenue_per_ticket"],
            [
                "Средняя выручка заказа, руб.",
                "Среднее число билетов",
                "Выручка на билет, руб.",
            ],
        ):
            bins = np.linspace(0, max(float(self.profiles[column].max()), 1), 36)
            for flag, label, color in [
                (False, "Один заказ", "#4477aa"),
                (True, "Два и более", "#ee7733"),
            ]:
                values = self.profiles.loc[self.profiles.is_two.eq(flag), column]
                if not values.empty:
                    axis.hist(
                        values,
                        bins=bins,
                        density=True,
                        alpha=0.5,
                        label=label,
                        color=color,
                    )
            axis.set(title=title, ylabel="Плотность")
            axis.legend()
        self._save_figure(figure, "purchase_patterns")

        weekdays = self.retention["first_weekday"].sort_index().copy()
        weekdays.index = weekdays.index.map(
            dict(enumerate(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]))
        )
        self._plot_segments(weekdays, "День первой покупки", "first_weekday")
        for name, matrix in self.correlations.items():
            target = matrix[["total_orders"]] if name == "numeric" else matrix
            figure = Figure(figsize=(6, 7) if name == "numeric" else (12, 9))
            axis = figure.subplots()
            sns.heatmap(
                target,
                annot=True,
                fmt=".3f",
                vmin=0,
                vmax=1,
                cmap="Blues",
                ax=axis,
            )
            axis.set_title("PhiK: " + name)
            self._save_figure(figure, "phik_" + name)
