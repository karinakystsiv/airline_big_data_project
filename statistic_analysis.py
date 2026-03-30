import logging
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType

logger = logging.getLogger(__name__)

def analyze_numerical_features(df: DataFrame) -> DataFrame:
    logger.info("=== Етап попередньої обробки: Аналіз числових ознак ===")

    # Всі числові колонки за їхнім типом даних
    numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)]

    logger.info(f"Знайдено {len(numeric_cols)} числових колонок.")
    logger.info(f"Перелік числових колонок: {numeric_cols}")

    if not numeric_cols:
        logger.warning("У DataFrame немає числових колонок для аналізу.")
        return df

    logger.info("Обчислення статистики...")

    stats_df = df.select(*numeric_cols).summary()

    logger.info("Статистика числових ознак:")
    stats_df.show()

    logger.info("=== Аналіз числових ознак завершено ===")

    return stats_df
