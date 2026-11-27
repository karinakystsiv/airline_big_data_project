import logging
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # без графічного інтерфейсу (Docker)
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator,
                                   RegressionEvaluator)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# ---------------------------- Метрики ----------------------------

def regression_metrics(predictions: DataFrame, label: str) -> Dict[str, float]:
    """RMSE, MAE, R² для регресійної моделі."""
    ev = RegressionEvaluator(labelCol=label, predictionCol="prediction")
    return {
        "RMSE": ev.evaluate(predictions, {ev.metricName: "rmse"}),
        "MAE": ev.evaluate(predictions, {ev.metricName: "mae"}),
        "R2": ev.evaluate(predictions, {ev.metricName: "r2"}),
    }


def classification_metrics(predictions: DataFrame, label: str) -> Dict[str, float]:
    """
    Accuracy, Precision, Recall, F1 для позитивного класу (рейс запізнився),
    зважений F1 по обох класах та ROC-AUC.
    """
    ev = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction")
    m = {
        "Accuracy": ev.evaluate(predictions, {ev.metricName: "accuracy"}),
        "Precision": ev.evaluate(predictions, {ev.metricName: "precisionByLabel", ev.metricLabel: 1.0}),
        "Recall": ev.evaluate(predictions, {ev.metricName: "recallByLabel", ev.metricLabel: 1.0}),
        "F1": ev.evaluate(predictions, {ev.metricName: "fMeasureByLabel", ev.metricLabel: 1.0}),
        "F1_weighted": ev.evaluate(predictions, {ev.metricName: "f1"}),
    }
    auc_ev = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction",
                                           metricName="areaUnderROC")
    m["ROC_AUC"] = auc_ev.evaluate(predictions)
    return m


def confusion_matrix(predictions: DataFrame, label: str) -> pd.DataFrame:
    cm = (predictions.groupBy(label, "prediction").count().toPandas()
          .pivot(index=label, columns="prediction", values="count").fillna(0).astype(int))
    cm.index = [f"факт {int(i)}" for i in cm.index]
    cm.columns = [f"прогноз {int(c)}" for c in cm.columns]
    return cm


# ---------------------------- Збереження ----------------------------

def save_metrics_table(spark, rows: List[Dict], path: str) -> pd.DataFrame:
    """Зберігає таблицю порівняння моделей у CSV (як і результати бізнес-питань)."""
    pdf = pd.DataFrame(rows)
    spark.createDataFrame(pdf).coalesce(1).write.mode("overwrite").csv(path, header=True)
    logger.info(f"Метрики збережено: {path}")
    return pdf


# ---------------------------- Графіки ----------------------------

def _savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"Збережено графік: {path}")


def plot_metric_comparison(pdf: pd.DataFrame, metrics: List[str], title: str, path: str):
    """Порівняння моделей за метриками на тестовій вибірці."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        ax.bar(pdf["model"], pdf[f"test_{m}"], color="steelblue")
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(pdf[f"test_{m}"]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title)
    _savefig(path)


def plot_train_vs_test(pdf: pd.DataFrame, metric: str, title: str, path: str):
    """Метрика на трейні та тесті поруч - видно перенавчання."""
    x = range(len(pdf))
    w = 0.38
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - w / 2 for i in x], pdf[f"train_{metric}"], w, label="train")
    plt.bar([i + w / 2 for i in x], pdf[f"test_{metric}"], w, label="test")
    plt.xticks(list(x), pdf["model"], rotation=20)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    _savefig(path)


def plot_curve(curves: Dict[str, List[float]], xlabel: str, ylabel: str, title: str, path: str):
    """Довільні криві навчання (історія функції втрат, втрати по ітераціях GBT тощо)."""
    plt.figure(figsize=(8, 4.5))
    for name, values in curves.items():
        plt.plot(range(1, len(values) + 1), values, marker=".", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(path)


def plot_validation_curve(param_values, train_scores, test_scores, param_name, metric, title, path):
    """Крива валідації: як змінюється якість зі зростанням складності моделі."""
    plt.figure(figsize=(8, 4.5))
    plt.plot(param_values, train_scores, marker="o", label="train")
    plt.plot(param_values, test_scores, marker="o", label="test")
    plt.xlabel(param_name)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    _savefig(path)


def plot_feature_importance(importances, feature_names: List[str], title: str, path: str, top: int = 15):
    pairs = sorted(zip(feature_names, importances.toArray()), key=lambda p: p[1], reverse=True)[:top]
    names, values = zip(*pairs)
    plt.figure(figsize=(8, 0.35 * len(names) + 1.5))
    plt.barh(list(reversed(names)), list(reversed(values)), color="seagreen")
    plt.title(title)
    plt.xlabel("Важливість")
    _savefig(path)
    return pairs


def plot_regression_diagnostics(predictions: DataFrame, label: str, title: str, path: str,
                                sample_size: int = 5000):
    """Прогноз vs факт та розподіл залишків для найкращого регресора."""
    total = predictions.count()
    frac = min(1.0, sample_size / max(total, 1))
    pdf = predictions.select(label, "prediction").sample(fraction=frac, seed=1).toPandas()
    residuals = pdf[label] - pdf["prediction"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(pdf[label], pdf["prediction"], s=4, alpha=0.3)
    lo, hi = pdf[label].min(), pdf[label].max()
    axes[0].plot([lo, hi], [lo, hi], color="red", lw=1)
    axes[0].set_xlabel("Фактична затримка, хв")
    axes[0].set_ylabel("Прогноз, хв")
    axes[0].set_title("Прогноз vs факт")
    axes[1].hist(residuals, bins=60, color="slateblue")
    axes[1].set_xlabel("Залишок (факт - прогноз), хв")
    axes[1].set_title("Розподіл залишків")
    fig.suptitle(title)
    _savefig(path)


def plot_confusion_matrix(cm: pd.DataFrame, title: str, path: str):
    plt.figure(figsize=(5, 4.2))
    plt.imshow(cm.values, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm.values[i, j]}", ha="center", va="center")
    plt.xticks(range(cm.shape[1]), cm.columns)
    plt.yticks(range(cm.shape[0]), cm.index)
    plt.title(title)
    plt.colorbar()
    _savefig(path)
