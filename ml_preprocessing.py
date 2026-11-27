import logging
import math
from typing import Dict, List, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# Поріг затримки прибуття, після якого рейс вважається запізнілим (стандарт US DOT - 15 хв)
DELAY_THRESHOLD_MIN = 15

# Цільові змінні
REGRESSION_LABEL = "ARR_DELAY"
CLASSIFICATION_LABEL = "IS_DELAYED"

# Ознаки для регресії: прогноз затримки прибуття вже ПІСЛЯ вильоту
# (затримка вильоту та час руління відомі на момент зльоту)
REGRESSION_NUMERIC = [
    "DEP_DELAY", "TAXI_OUT", "DISTANCE", "CRS_ELAPSED_TIME",
    "HOUR_SIN", "HOUR_COS", "MONTH_SIN", "MONTH_COS", "IS_WEEKEND",
    "ORIGIN_TE", "DEST_TE",
]

# Ознаки для класифікації: тільки те, що відомо ДО вильоту (без витоку цільової змінної)
CLASSIFICATION_NUMERIC = [
    "DISTANCE", "CRS_ELAPSED_TIME",
    "HOUR_SIN", "HOUR_COS", "MONTH_SIN", "MONTH_COS", "IS_WEEKEND",
    "ORIGIN_TE", "DEST_TE",
]

# Категоріальна ознака з невеликою кількістю значень - кодуємо через one-hot
CATEGORICAL = ["OP_CARRIER"]

# Категоріальні ознаки з сотнями значень - кодуємо через target encoding
HIGH_CARDINALITY = ["ORIGIN", "DEST"]


def select_flights_for_ml(df: DataFrame, max_rows: int = None, seed: int = 42) -> DataFrame:
    """Відбір рядків, придатних для навчання моделей, та створення цільових змінних."""
    logger.info("=== ML: відбір даних для моделювання ===")

    # Для скасованих і перенаправлених рейсів затримка прибуття не визначена
    df = df.filter((F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0))

    needed = ["ARR_DELAY", "DEP_DELAY", "TAXI_OUT", "DISTANCE", "CRS_ELAPSED_TIME",
              "DEP_HOUR", "MONTH", "DAY_OF_WEEK", "OP_CARRIER", "ORIGIN", "DEST"]
    df = df.dropna(subset=needed)

    # Видалення екстремальних викидів затримки (верхні 0.5%) - вони сильно псують RMSE
    upper = df.approxQuantile(REGRESSION_LABEL, [0.995], 0.001)[0]
    df = df.filter((F.col(REGRESSION_LABEL) >= -90) & (F.col(REGRESSION_LABEL) <= upper))
    logger.info(f"Відсічено затримки прибуття поза межами [-90; {upper:.0f}] хв.")

    if max_rows is not None:
        total = df.count()
        if total > max_rows:
            df = df.sample(fraction=max_rows / total, seed=seed)
            logger.info(f"Взято підвибірку ~{max_rows} рядків із {total}.")

    # Бінарна цільова змінна для класифікації
    df = df.withColumn(
        CLASSIFICATION_LABEL,
        (F.col(REGRESSION_LABEL) >= DELAY_THRESHOLD_MIN).cast("double"),
    )
    df = df.withColumn(REGRESSION_LABEL, F.col(REGRESSION_LABEL).cast("double"))
    return df


def add_engineered_features(df: DataFrame) -> DataFrame:
    """Циклічне кодування часу, ознака вихідного дня, приведення типів."""
    two_pi = 2 * math.pi
    df = (
        df
        # Година та місяць - циклічні величини: 23:00 близько до 00:00, грудень близько до січня
        .withColumn("HOUR_SIN", F.sin(two_pi * F.col("DEP_HOUR") / 24))
        .withColumn("HOUR_COS", F.cos(two_pi * F.col("DEP_HOUR") / 24))
        .withColumn("MONTH_SIN", F.sin(two_pi * (F.col("MONTH") - 1) / 12))
        .withColumn("MONTH_COS", F.cos(two_pi * (F.col("MONTH") - 1) / 12))
        # У Spark dayofweek: 1 = неділя, 7 = субота
        .withColumn("IS_WEEKEND", F.col("DAY_OF_WEEK").isin(1, 7).cast("double"))
    )
    for c in ["DEP_DELAY", "TAXI_OUT", "DISTANCE", "CRS_ELAPSED_TIME"]:
        df = df.withColumn(c, F.col(c).cast("double"))
    return df


def fit_target_encoding(train: DataFrame, cols: List[str], label: str,
                        smoothing: float = 50.0) -> Dict[str, DataFrame]:
    """
    Target encoding зі згладжуванням: кожен аеропорт замінюється середнім значенням
    цільової змінної для нього, «підтягнутим» до глобального середнього, якщо рейсів мало.
    Рахується ТІЛЬКИ на тренувальній вибірці, щоб не було витоку з тестової.
    """
    global_mean = train.agg(F.avg(label)).first()[0]
    encodings = {}
    for c in cols:
        stats = train.groupBy(c).agg(F.count("*").alias("n"), F.avg(label).alias("m"))
        encodings[c] = stats.select(
            c,
            ((F.col("n") * F.col("m") + smoothing * global_mean) /
             (F.col("n") + smoothing)).alias(f"{c}_TE"),
        )
    encodings["__global__"] = global_mean
    return encodings


def apply_target_encoding(df: DataFrame, encodings: Dict) -> DataFrame:
    global_mean = encodings["__global__"]
    for c, enc in encodings.items():
        if c == "__global__":
            continue
        # Невідомі аеропорти (є в тесті, нема в трейні) отримують глобальне середнє
        df = df.join(F.broadcast(enc), on=c, how="left").withColumn(
            f"{c}_TE", F.coalesce(F.col(f"{c}_TE"), F.lit(global_mean))
        )
    return df


def build_feature_pipeline(numeric_cols: List[str]) -> Pipeline:
    """StringIndexer -> OneHotEncoder -> VectorAssembler -> StandardScaler."""
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_IDX", handleInvalid="keep")
                for c in CATEGORICAL]
    encoder = OneHotEncoder(inputCols=[f"{c}_IDX" for c in CATEGORICAL],
                            outputCols=[f"{c}_OHE" for c in CATEGORICAL])
    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{c}_OHE" for c in CATEGORICAL],
        outputCol="raw_features",
        handleInvalid="skip",
    )
    # Масштабування потрібне лінійним моделям; деревам воно не шкодить
    scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                            withMean=False, withStd=True)
    return Pipeline(stages=indexers + [encoder, assembler, scaler])


def add_class_weights(train: DataFrame, test: DataFrame, label: str) -> Tuple[DataFrame, DataFrame, float]:
    """
    Ваги класів для боротьби з дисбалансом: рідший клас отримує більшу вагу.
    Ваги рахуються по трейну; у тест додаються ті самі ваги лише для того, щоб
    можна було порахувати криву втрат GBT (метрики якості їх не використовують).
    """
    pos_ratio = train.agg(F.avg(label)).first()[0]
    w_pos = 0.5 / pos_ratio
    w_neg = 0.5 / (1 - pos_ratio)
    weight = F.when(F.col(label) == 1.0, w_pos).otherwise(w_neg)
    train = train.withColumn("class_weight", weight)
    test = test.withColumn("class_weight", weight)
    logger.info(f"Частка запізнілих рейсів у трейні: {pos_ratio:.3f}; "
                f"ваги класів: 0 -> {w_neg:.3f}, 1 -> {w_pos:.3f}")
    return train, test, pos_ratio


def get_feature_names(pipeline_model: PipelineModel, numeric_cols: List[str]) -> List[str]:
    """Імена ознак у порядку, в якому вони лежать у векторі features (для важливості ознак)."""
    names = list(numeric_cols)
    for stage in pipeline_model.stages:
        if hasattr(stage, "labels") and stage.getOutputCol().endswith("_IDX"):
            base = stage.getInputCol()
            # OneHotEncoder за замовчуванням відкидає останню категорію (dropLast=True);
            # handleInvalid="keep" додає ще одну категорію для невідомих значень
            labels = list(stage.labels) + ["__unknown__"]
            names += [f"{base}={lab}" for lab in labels[:-1]]
    return names


def prepare_datasets(df: DataFrame, task: str, max_rows: int = None, seed: int = 42):
    """
    Повна підготовка даних для задачі task ('regression' або 'classification').
    Повертає (train, test, feature_names, extra_info).
    """
    assert task in ("regression", "classification")
    label = REGRESSION_LABEL if task == "regression" else CLASSIFICATION_LABEL
    numeric = REGRESSION_NUMERIC if task == "regression" else CLASSIFICATION_NUMERIC

    data = select_flights_for_ml(df, max_rows=max_rows, seed=seed)
    data = add_engineered_features(data)

    # Розбиття 80/20 ДО будь-якого навчання на даних (кодування, масштабування)
    train, test = data.randomSplit([0.8, 0.2], seed=seed)

    encodings = fit_target_encoding(train, HIGH_CARDINALITY, label)
    train = apply_target_encoding(train, encodings)
    test = apply_target_encoding(test, encodings)

    pipeline_model = build_feature_pipeline(numeric).fit(train)
    keep = ["features", label]
    train = pipeline_model.transform(train).select(*keep)
    test = pipeline_model.transform(test).select(*keep)

    info = {}
    if task == "classification":
        train, test, info["positive_ratio"] = add_class_weights(train, test, label)

    train = train.cache()
    test = test.cache()
    info["n_train"] = train.count()
    info["n_test"] = test.count()
    logger.info(f"[{task}] train: {info['n_train']} рядків, test: {info['n_test']} рядків.")

    return train, test, get_feature_names(pipeline_model, numeric), info
