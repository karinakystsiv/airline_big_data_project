import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

logger = logging.getLogger(__name__)

def _log_null_stats(df: DataFrame, stage: str) -> None:
    logger.info(f"--- Статистика пропусків ({stage}) ---")
    total = df.count()

    null_exprs = [
        F.sum(F.when(F.isnull(c), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    null_counts = df.select(null_exprs).collect()[0]

    has_nulls = False
    for col_name in df.columns:
        nulls = null_counts[col_name]
        if nulls > 0:
            pct = (nulls / total) * 100
            logger.info(f"  {col_name}: {nulls} пропусків ({pct:.2f}%)")
            has_nulls = True

    if not has_nulls:
        logger.info("  Пропусків не знайдено.")


def _parse_hhmm_column(df: DataFrame, col_name: str) -> DataFrame:
    parsed_col = f"{col_name}_PARSED"

    padded = F.lpad(
        F.col(col_name).cast(IntegerType()).cast("string"), 4, "0"
    )

    df = df.withColumn(
        parsed_col,
        F.when(
            F.col(col_name).isNotNull(),
            F.concat(
                F.substring(padded, 1, 2),
                F.lit(":"),
                F.substring(padded, 3, 2),
            ),
        ),
    )
    return df


def _build_timestamp(df: DataFrame, date_col: str, parsed_col: str) -> DataFrame:
    ts_col = parsed_col.replace("_PARSED", "_TS")

    df = df.withColumn(
        ts_col,
        F.when(
            F.col(parsed_col).isNotNull(),
            F.to_timestamp(
                F.concat(
                    F.col(date_col).cast("string"),
                    F.lit(" "),
                    F.col(parsed_col),
                ),
                "yyyy-MM-dd HH:mm",
            ),
        ),
    )
    return df



def handle_missing_values(df: DataFrame) -> DataFrame:

    logger.info("=== Обробка пропущених значень ===")
    _log_null_stats(df, "до обробки")

    # 1. Затримки: null -> 0.0
    delay_cols = [
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ]
    for c in delay_cols:
        if c in df.columns:
            df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))
    logger.info(f"Заповнено нулями {len(delay_cols)} колонок затримок.")

    if "CANCELLATION_CODE" in df.columns:
        df = df.withColumn(
            "CANCELLATION_CODE",
            F.coalesce(F.col("CANCELLATION_CODE"), F.lit("N/A")),
        )
        logger.info("CANCELLATION_CODE: пропуски заповнено значенням 'N/A'.")

    # 3. Видалення рядків без критичних полів
    critical_cols = ["FL_DATE", "OP_CARRIER", "ORIGIN", "DEST"]
    before = df.count()
    df = df.dropna(subset=critical_cols)
    after = df.count()
    dropped = before - after
    logger.info(
        f"Видалено {dropped} рядків з пропусками у критичних полях {critical_cols}."
        if dropped > 0
        else "Рядків з пропусками у критичних полях не знайдено."
    )

    _log_null_stats(df, "після обробки")
    logger.info("=== Обробка пропущених значень завершена ===")
    return df


def parse_datetime_features(df: DataFrame) -> DataFrame:
    logger.info("=== Парсинг часових ознак ===")

    # 1. Парсинг HHMM -> 'HH:mm'
    time_cols = [
        "CRS_DEP_TIME", "DEP_TIME",
        "CRS_ARR_TIME", "ARR_TIME",
        "WHEELS_OFF", "WHEELS_ON",
    ]
    for c in time_cols:
        if c in df.columns:
            df = _parse_hhmm_column(df, c)
            logger.info(f"  {c} -> {c}_PARSED")

    # 2. Побудова Timestamp-ів для ключових пар (дата + час)
    timestamp_cols = ["CRS_DEP_TIME", "DEP_TIME", "CRS_ARR_TIME", "ARR_TIME"]
    for c in timestamp_cols:
        parsed = f"{c}_PARSED"
        if parsed in df.columns:
            df = _build_timestamp(df, "FL_DATE", parsed)
            logger.info(f"  {parsed} -> {c}_TS")

    # 3. Додаткові часові ознаки
    df = (
        df
        .withColumn("YEAR", F.year("FL_DATE"))
        .withColumn("MONTH", F.month("FL_DATE"))
        .withColumn("DAY_OF_WEEK", F.dayofweek("FL_DATE"))
        .withColumn("DEP_HOUR",
                     F.substring(F.col("CRS_DEP_TIME_PARSED"), 1, 2).cast(IntegerType()))
    )
    logger.info("Додано ознаки: YEAR, MONTH, DAY_OF_WEEK, DEP_HOUR.")

    logger.info("=== Парсинг часових ознак завершено ===")
    return df


def handle_duplicates(df: DataFrame) -> DataFrame:
    logger.info("=== Аналіз та видалення дублікатів ===")

    total = df.count()
    logger.info(f"Загальна кількість рядків: {total}")

    # 1. Повні дублікати (всі колонки ідентичні)
    df_no_full_dupes = df.dropDuplicates()
    after_full = df_no_full_dupes.count()
    full_dupes = total - after_full
    logger.info(f"Повних дублікатів: {full_dupes} ({(full_dupes / total) * 100:.2f}%)")

    # 2. Часткові дублікати за ключовими полями рейсу
    key_cols = ["FL_DATE", "OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME"]
    present_keys = [c for c in key_cols if c in df_no_full_dupes.columns]

    df_deduped = df_no_full_dupes.dropDuplicates(subset=present_keys)
    after_partial = df_deduped.count()
    partial_dupes = after_full - after_partial
    logger.info(
        f"Часткових дублікатів (за {present_keys}): {partial_dupes} ({(partial_dupes / total) * 100:.2f}%)"
    )

    total_removed = total - after_partial
    logger.info(f"Всього видалено дублікатів: {total_removed} з {total} рядків")
    logger.info(f"Залишилось рядків: {after_partial}")
    logger.info("=== Аналіз дублікатів завершено ===")

    return df_deduped


def preprocess(df: DataFrame) -> DataFrame:
    df = handle_duplicates(df)
    df = handle_missing_values(df)
    df = parse_datetime_features(df)
    return df