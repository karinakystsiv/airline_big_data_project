import logging
import time

from pyspark.ml.classification import (DecisionTreeClassifier, GBTClassifier,
                                       LogisticRegression, RandomForestClassifier)
from pyspark.sql import functions as F

import model_evaluation as ev
from ml_preprocessing import CLASSIFICATION_LABEL, prepare_datasets

logger = logging.getLogger(__name__)

LABEL = CLASSIFICATION_LABEL
WEIGHT = "class_weight"


def get_classifiers(seed: int = 42):
    """Чотири класифікатори; усі враховують ваги класів через weightCol."""
    return {
        "LogisticRegression": LogisticRegression(
            featuresCol="features", labelCol=LABEL, weightCol=WEIGHT,
            maxIter=50, regParam=0.01, elasticNetParam=0.0,
        ),
        "DecisionTree": DecisionTreeClassifier(
            featuresCol="features", labelCol=LABEL, weightCol=WEIGHT,
            maxDepth=8, minInstancesPerNode=20, seed=seed,
        ),
        "RandomForest": RandomForestClassifier(
            featuresCol="features", labelCol=LABEL, weightCol=WEIGHT,
            numTrees=40, maxDepth=8, subsamplingRate=0.8, featureSubsetStrategy="sqrt", seed=seed,
        ),
        "GBT": GBTClassifier(
            featuresCol="features", labelCol=LABEL, weightCol=WEIGHT,
            maxIter=40, maxDepth=5, stepSize=0.1, subsamplingRate=0.8, seed=seed,
        ),
    }


def _analyze_training(name, model, train, test, out_dir):
    if name == "LogisticRegression":
        history = model.summary.objectiveHistory
        logger.info(f"[{name}] ітерацій оптимізатора: {model.summary.totalIterations}, "
                    f"log-loss: {history[0]:.4f} -> {history[-1]:.4f}")
        ev.plot_curve({"objective": history}, "Ітерація", "Функція втрат (log-loss + L2)",
                      "LogisticRegression: збіжність оптимізатора",
                      f"{out_dir}/plots/clf_logreg_objective_history.png")

    if name == "GBT":
        train_loss = model.evaluateEachIteration(train)
        test_loss = model.evaluateEachIteration(test)
        best_iter = int(min(range(len(test_loss)), key=lambda i: test_loss[i])) + 1
        logger.info(f"[{name}] log-loss на тесті мінімальна після {best_iter} дерев "
                    f"із {len(test_loss)}.")
        ev.plot_curve({"train": train_loss, "test": test_loss}, "Кількість дерев", "Log-loss",
                      "GBTClassifier: крива навчання", f"{out_dir}/plots/clf_gbt_learning_curve.png")


def validation_curve_tree_depth(train, test, out_dir, depths=(2, 4, 6, 8, 10, 12), seed=42):
    train_f1, test_f1 = [], []
    for d in depths:
        m = DecisionTreeClassifier(featuresCol="features", labelCol=LABEL, weightCol=WEIGHT,
                                   maxDepth=d, minInstancesPerNode=20, seed=seed).fit(train)
        train_f1.append(ev.classification_metrics(m.transform(train), LABEL)["F1"])
        test_f1.append(ev.classification_metrics(m.transform(test), LABEL)["F1"])
        logger.info(f"  maxDepth={d}: F1 train={train_f1[-1]:.4f}, test={test_f1[-1]:.4f}")
    ev.plot_validation_curve(list(depths), train_f1, test_f1, "maxDepth", "F1 (клас 1)",
                             "DecisionTreeClassifier: крива валідації",
                             f"{out_dir}/plots/clf_tree_validation_curve.png")
    return list(depths), train_f1, test_f1


def run_classification(spark, df, out_dir="output/data_analysis", max_rows=None, seed=42):
    logger.info("========== КЛАСИФІКАЦІЯ: чи запізниться рейс на >= 15 хв (IS_DELAYED) ==========")
    train, test, feature_names, info = prepare_datasets(df, "classification", max_rows=max_rows, seed=seed)

    # Baseline: завжди прогнозуємо більший клас «вчасно». Висока accuracy, але recall = 0
    base_pred = test.withColumn("prediction", F.lit(0.0))
    base_acc = base_pred.filter(F.col(LABEL) == 0.0).count() / info["n_test"]
    logger.info(f"Baseline (завжди 'вчасно'): Accuracy={base_acc:.4f}, Recall=0")
    rows = [{"model": "Baseline(majority)", "train_time_sec": 0.0,
             "train_Accuracy": None, "train_Precision": None, "train_Recall": None,
             "train_F1": None, "train_F1_weighted": None, "train_ROC_AUC": None,
             "test_Accuracy": base_acc, "test_Precision": 0.0, "test_Recall": 0.0,
             "test_F1": 0.0, "test_F1_weighted": None, "test_ROC_AUC": 0.5}]
    fitted = {}

    for name, estimator in get_classifiers(seed).items():
        logger.info(f"--- Навчання {name} ---")
        start = time.time()
        model = estimator.fit(train)
        elapsed = time.time() - start

        train_m = ev.classification_metrics(model.transform(train), LABEL)
        test_m = ev.classification_metrics(model.transform(test), LABEL)
        logger.info(f"[{name}] {elapsed:.1f} c | train F1={train_m['F1']:.4f} AUC={train_m['ROC_AUC']:.4f} "
                    f"| test Acc={test_m['Accuracy']:.4f} P={test_m['Precision']:.4f} "
                    f"R={test_m['Recall']:.4f} F1={test_m['F1']:.4f} AUC={test_m['ROC_AUC']:.4f}")

        _analyze_training(name, model, train, test, out_dir)
        rows.append({"model": name, "train_time_sec": round(elapsed, 2),
                     **{f"train_{k}": v for k, v in train_m.items()},
                     **{f"test_{k}": v for k, v in test_m.items()}})
        fitted[name] = model

    for name in ("RandomForest", "GBT"):
        top = ev.plot_feature_importance(fitted[name].featureImportances, feature_names,
                                         f"{name}Classifier: важливість ознак",
                                         f"{out_dir}/plots/clf_{name.lower()}_feature_importance.png")
        logger.info(f"[{name}] топ-5 ознак: " + ", ".join(f"{n} ({v:.3f})" for n, v in top[:5]))

    logger.info("--- Крива валідації для глибини дерева ---")
    validation_curve_tree_depth(train, test, out_dir, seed=seed)

    pdf = ev.save_metrics_table(spark, rows, f"{out_dir}/classification_metrics")
    models_only = pdf[~pdf["model"].str.startswith("Baseline")].reset_index(drop=True)
    ev.plot_metric_comparison(models_only, ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
                              "Класифікація: порівняння моделей (test)", f"{out_dir}/plots/clf_comparison.png")
    ev.plot_train_vs_test(models_only, "F1", "Класифікація: F1 на train і test",
                          f"{out_dir}/plots/clf_train_vs_test_f1.png")

    # Найкраща модель - за F1 позитивного класу (accuracy при дисбалансі вводить в оману)
    best = models_only.sort_values("test_F1", ascending=False).iloc[0]["model"]
    cm = ev.confusion_matrix(fitted[best].transform(test), LABEL)
    logger.info(f"Найкращий класифікатор за F1 на тесті: {best}\nМатриця помилок:\n{cm}")
    ev.plot_confusion_matrix(cm, f"{best}: матриця помилок (test)",
                             f"{out_dir}/plots/clf_best_confusion_matrix.png")

    print("\nКЛАСИФІКАЦІЯ - порівняння моделей:")
    print(pdf.round(4).to_string(index=False))

    train.unpersist()
    test.unpersist()
    return pdf, info
