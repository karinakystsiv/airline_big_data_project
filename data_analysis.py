import logging

from pyspark.sql import DataFrame, SparkSession

from classification_models import run_classification
from regression_models import run_regression

logger = logging.getLogger(__name__)


def run_data_analysis(spark: SparkSession, df: DataFrame,
                      output_path: str = "output/data_analysis",
                      max_rows: int = 500_000, seed: int = 42):
    """
    Етап аналізу даних: регресія (скільки хвилин затримки прибуття)
    та класифікація (чи запізниться рейс на 15+ хв).
    max_rows обмежує кількість рядків для навчання, щоб пайплайн працював локально.
    """
    logger.info("=============== ЕТАП АНАЛІЗУ ДАНИХ (ML) ===============")
    df = df.cache()

    reg_results, _ = run_regression(spark, df, out_dir=output_path, max_rows=max_rows, seed=seed)
    clf_results, _ = run_classification(spark, df, out_dir=output_path, max_rows=max_rows, seed=seed)

    df.unpersist()
    logger.info(f"Етап аналізу даних завершено. Метрики та графіки: {output_path}/")
    return reg_results, clf_results
