import logging
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

logger = logging.getLogger(__name__)

def validate_dataframe(df: DataFrame, expected_schema: StructType) -> bool:
    logger.info("=== Початок валідації даних ===")

    row_count = df.count()
    if row_count == 0:
        logger.error("DataFrame порожній.")
        return False
    else:
        logger.info(f"Валідація рядків пройдена: знайдено {row_count} рядків.")

    if df.schema != expected_schema:
        logger.error("Зчитана схема не збігається з очікуваною")
        return False
    else:
        logger.info("Валідація схеми пройдена: структура даних відповідає очікуванням.")

    logger.info("=== Валідація успішно завершена ===")
    return True