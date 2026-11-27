import logging
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as F
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

    # Графік 1: Розподіл рейсів за авіакомпаніями (Top 10)

    carrier_counts = df.groupBy("OP_CARRIER").count().orderBy(F.desc("count")).limit(10).toPandas()
    plt.figure(figsize=(10, 6))
    sns.barplot(x="count", y="OP_CARRIER", data=carrier_counts, palette="viridis")
    plt.title("Топ-10 авіакомпаній за кількістю рейсів")
    plt.savefig("top_carriers.png")
    plt.close()
    logger.info("Збережено графік: top_carriers.png")

    # Графік 2: Співвідношення скасованих рейсів (Pie Chart)

    cancelled_counts = df.groupBy("CANCELLED").count().orderBy("CANCELLED").toPandas()

    plt.figure(figsize=(7, 7))
    plt.pie(
        cancelled_counts["count"],

        labels=["Виконано (0)", "Скасовано (1)"],
        autopct='%1.1f%%',
        colors=['skyblue', 'red'],
        startangle=140
    )
    plt.title("Відсоток скасованих рейсів")
    plt.savefig("cancelled_pie.png")
    plt.close()
    logger.info("Збережено графік: cancelled_pie.png")

    # Графік 3: Гістограма затримок вильоту (DEP_DELAY)

    delay_data = df.select("DEP_DELAY").filter((F.col("DEP_DELAY") > 0) & (F.col("DEP_DELAY") < 200)).sample(
        0.01).toPandas()
    plt.figure(figsize=(10, 6))
    sns.histplot(delay_data["DEP_DELAY"], bins=30, kde=True, color="orange")
    plt.title("Розподіл затримок вильоту (до 200 хв)")
    plt.xlabel("Затримка (хвилини)")
    plt.savefig("delay_distribution.png")
    plt.close()
    logger.info("Збережено графік: delay_distribution.png")

    return stats_df
