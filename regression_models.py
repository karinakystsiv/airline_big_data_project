import logging
import time

from pyspark.ml.regression import (DecisionTreeRegressor, GBTRegressor,
                                   LinearRegression, RandomForestRegressor)

import model_evaluation as ev
from ml_preprocessing import REGRESSION_LABEL, prepare_datasets

logger = logging.getLogger(__name__)

LABEL = REGRESSION_LABEL


def get_regressors(seed: int = 42):
    """Чотири алгоритми регресії різної природи: лінійний, одне дерево, бегінг, бустинг."""
    return {
        "LinearRegression": LinearRegression(
            featuresCol="features", labelCol=LABEL,
            maxIter=50, regParam=0.01, elasticNetParam=0.0, solver="l-bfgs",  # L2-регуляризація
        ),
        "DecisionTree": DecisionTreeRegressor(
            featuresCol="features", labelCol=LABEL, maxDepth=8, minInstancesPerNode=20, seed=seed,
        ),
        "RandomForest": RandomForestRegressor(
            featuresCol="features", labelCol=LABEL, numTrees=40, maxDepth=8,
            subsamplingRate=0.8, featureSubsetStrategy="onethird", seed=seed,
        ),
        "GBT": GBTRegressor(
            featuresCol="features", labelCol=LABEL, maxIter=40, maxDepth=5,
            stepSize=0.1, subsamplingRate=0.8, seed=seed,
        ),
    }


def _analyze_training(name, model, train, test, out_dir):
    """Аналіз процесу навчання, специфічний для кожного типу моделі."""
    if name == "LinearRegression":
        history = model.summary.objectiveHistory
        logger.info(f"[{name}] ітерацій оптимізатора: {model.summary.totalIterations}, "
                    f"втрати: {history[0]:.4f} -> {history[-1]:.4f}")
        if len(history) > 1:
            ev.plot_curve({"objective": history}, "Ітерація", "Функція втрат",
                          "LinearRegression: збіжність оптимізатора",
                          f"{out_dir}/plots/reg_linear_objective_history.png")

    if name == "GBT":
        # Помилка після кожного нового дерева на трейні й тесті - крива навчання бустингу
        train_loss = model.evaluateEachIteration(train, "squared")
        test_loss = model.evaluateEachIteration(test, "squared")
        best_iter = int(min(range(len(test_loss)), key=lambda i: test_loss[i])) + 1
        logger.info(f"[{name}] MSE на тесті мінімальна після {best_iter} дерев "
                    f"із {len(test_loss)}.")
        ev.plot_curve({"train": train_loss, "test": test_loss}, "Кількість дерев", "MSE",
                      "GBTRegressor: крива навчання", f"{out_dir}/plots/reg_gbt_learning_curve.png")


def validation_curve_tree_depth(train, test, out_dir, depths=(2, 4, 6, 8, 10, 12), seed=42):
    """Як глибина дерева впливає на якість: недонавчання -> оптимум -> перенавчання."""
    train_rmse, test_rmse = [], []
    for d in depths:
        m = DecisionTreeRegressor(featuresCol="features", labelCol=LABEL,
                                  maxDepth=d, minInstancesPerNode=20, seed=seed).fit(train)
        train_rmse.append(ev.regression_metrics(m.transform(train), LABEL)["RMSE"])
        test_rmse.append(ev.regression_metrics(m.transform(test), LABEL)["RMSE"])
        logger.info(f"  maxDepth={d}: RMSE train={train_rmse[-1]:.3f}, test={test_rmse[-1]:.3f}")
    ev.plot_validation_curve(list(depths), train_rmse, test_rmse, "maxDepth", "RMSE",
                             "DecisionTreeRegressor: крива валідації",
                             f"{out_dir}/plots/reg_tree_validation_curve.png")
    return list(depths), train_rmse, test_rmse


def run_regression(spark, df, out_dir="output/data_analysis", max_rows=None, seed=42):
    logger.info("========== РЕГРЕСІЯ: прогноз затримки прибуття (ARR_DELAY, хв) ==========")
    train, test, feature_names, info = prepare_datasets(df, "regression", max_rows=max_rows, seed=seed)

    # Базова модель-орієнтир: завжди прогнозує середнє з трейну (R² ≈ 0)
    mean_delay = train.agg({LABEL: "avg"}).first()[0]
    from pyspark.sql import functions as F
    baseline = ev.regression_metrics(test.withColumn("prediction", F.lit(mean_delay)), LABEL)
    logger.info(f"Baseline (середнє = {mean_delay:.2f} хв): RMSE={baseline['RMSE']:.3f}")

    rows = [{"model": "Baseline(mean)", "train_time_sec": 0.0,
             **{f"train_{k}": v for k, v in baseline.items()},
             **{f"test_{k}": v for k, v in baseline.items()}}]
    fitted = {}

    for name, estimator in get_regressors(seed).items():
        logger.info(f"--- Навчання {name} ---")
        start = time.time()
        model = estimator.fit(train)
        elapsed = time.time() - start

        train_m = ev.regression_metrics(model.transform(train), LABEL)
        test_m = ev.regression_metrics(model.transform(test), LABEL)
        logger.info(f"[{name}] {elapsed:.1f} c | train RMSE={train_m['RMSE']:.3f} R2={train_m['R2']:.4f} "
                    f"| test RMSE={test_m['RMSE']:.3f} R2={test_m['R2']:.4f}")

        _analyze_training(name, model, train, test, out_dir)
        rows.append({"model": name, "train_time_sec": round(elapsed, 2),
                     **{f"train_{k}": v for k, v in train_m.items()},
                     **{f"test_{k}": v for k, v in test_m.items()}})
        fitted[name] = model

    # Коефіцієнти лінійної моделі (ознаки стандартизовані, тому їх можна порівнювати між собою)
    lin = fitted["LinearRegression"]
    coefs = sorted(zip(feature_names, lin.coefficients.toArray()), key=lambda p: abs(p[1]), reverse=True)
    logger.info("Найбільші за модулем коефіцієнти LinearRegression: " +
                ", ".join(f"{n}={c:.2f}" for n, c in coefs[:6]))

    for name in ("RandomForest", "GBT"):
        top = ev.plot_feature_importance(fitted[name].featureImportances, feature_names,
                                         f"{name}Regressor: важливість ознак",
                                         f"{out_dir}/plots/reg_{name.lower()}_feature_importance.png")
        logger.info(f"[{name}] топ-5 ознак: " + ", ".join(f"{n} ({v:.3f})" for n, v in top[:5]))

    logger.info("--- Крива валідації для глибини дерева ---")
    validation_curve_tree_depth(train, test, out_dir, seed=seed)

    pdf = ev.save_metrics_table(spark, rows, f"{out_dir}/regression_metrics")
    models_only = pdf[pdf["model"] != "Baseline(mean)"].reset_index(drop=True)
    ev.plot_metric_comparison(models_only, ["RMSE", "MAE", "R2"],
                              "Регресія: порівняння моделей (test)", f"{out_dir}/plots/reg_comparison.png")
    ev.plot_train_vs_test(models_only, "RMSE", "Регресія: RMSE на train і test",
                          f"{out_dir}/plots/reg_train_vs_test_rmse.png")

    best = models_only.sort_values("test_RMSE").iloc[0]["model"]
    logger.info(f"Найкраща регресійна модель за RMSE на тесті: {best}")
    ev.plot_regression_diagnostics(fitted[best].transform(test), LABEL,
                                   f"{best}: діагностика залишків",
                                   f"{out_dir}/plots/reg_best_diagnostics.png")

    print("\nРЕГРЕСІЯ - порівняння моделей:")
    print(pdf.round(4).to_string(index=False))

    train.unpersist()
    test.unpersist()
    return pdf, info
