"""
Elite Customer Churn Prediction Model — XGBoost + Hyperparameter Tuning
========================================================================
Includes:
- Time-safe features
- Categorical encoding
- XGBoost classifier
- Cross validation tuning
- Full metrics

Author: Data Engineering Portfolio Project
Date: February 2026
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import logging

from xgboost.spark import SparkXGBClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_churn_model(spark, data_path):

    logger.info("="*100)
    logger.info(" ELITE CUSTOMER CHURN MODEL — XGBOOST + HYPERPARAMETER TUNING")
    logger.info("="*100)

    train_df = spark.read.parquet(f"{data_path}/churn_train_v1")
    test_df  = spark.read.parquet(f"{data_path}/churn_test_v1")

    logger.info(f" Train customers: {train_df.count():,}")
    logger.info(f" Test customers:  {test_df.count():,}")

    numeric_features = [
        "total_purchases", "total_spend", "avg_order_value",
        "unique_products", "customer_tenure_days", "purchase_frequency",
        "days_since_last_purchase", "avg_days_between_purchases",
        "std_days_between_purchases", "online_ratio", "weekend_ratio"
    ]

    categorical_features = ["club_member_status", "fashion_news_frequency"]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_features
    ]

    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical_features],
        outputCols=[f"{c}_ohe" for c in categorical_features]
    )

    feature_cols = numeric_features + [f"{c}_ohe" for c in categorical_features]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    xgb = SparkXGBClassifier(
        label_col="is_churned",
        features_col="features",
        objective="binary:logistic",
        eval_metric="auc",
        num_workers=6,
        seed=42,
        missing=0.0
    )

    pipeline = Pipeline(stages=indexers + [encoder, assembler, xgb])

    paramGrid = ParamGridBuilder() \
        .addGrid(xgb.max_depth, [4, 6, 8]) \
        .addGrid(xgb.n_estimators, [150, 300]) \
        .addGrid(xgb.learning_rate, [0.03, 0.05, 0.1]) \
        .addGrid(xgb.subsample, [0.8, 1.0]) \
        .addGrid(xgb.colsample_bytree, [0.8, 1.0]) \
        .build()

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="is_churned",
        metricName="areaUnderROC"
    )

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator_auc,
        numFolds=3,
        parallelism=6,
        seed=42
    )

    logger.info("\n Training XGBoost model with hyperparameter tuning (this will take time)...")

    cv_model = crossval.fit(train_df)

    logger.info(" Hyperparameter tuning completed!")

    predictions = cv_model.transform(test_df)

    auc = evaluator_auc.evaluate(predictions)

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="is_churned", predictionCol="prediction", metricName="accuracy"
    )
    acc = evaluator_acc.evaluate(predictions)

    evaluator_prec = MulticlassClassificationEvaluator(
        labelCol="is_churned", predictionCol="prediction", metricName="weightedPrecision"
    )
    precision = evaluator_prec.evaluate(predictions)

    evaluator_rec = MulticlassClassificationEvaluator(
        labelCol="is_churned", predictionCol="prediction", metricName="weightedRecall"
    )
    recall = evaluator_rec.evaluate(predictions)

    logger.info("\n" + "="*100)
    logger.info(" FINAL XGBOOST MODEL PERFORMANCE")
    logger.info("="*100)
    logger.info(f"   AUC-ROC:   {auc:.4f}")
    logger.info(f"   Accuracy:  {acc:.4f}")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall:    {recall:.4f}")
    logger.info("="*100)

    best_xgb = cv_model.bestModel.stages[-1]
    logger.info("\n BEST MODEL PARAMETERS:")
    logger.info(f"   maxDepth: {best_xgb.getOrDefault('max_depth')}")
    logger.info(f"   nEstimators: {best_xgb.getOrDefault('n_estimators')}")
    logger.info(f"   learningRate: {best_xgb.getOrDefault('learning_rate')}")
    logger.info(f"   subsample: {best_xgb.getOrDefault('subsample')}")
    logger.info(f"   colsampleBytree: {best_xgb.getOrDefault('colsample_bytree')}")

    return cv_model


def main():
    import sys, os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    from scripts.etl.spark_config import get_spark_session, stop_spark_session, PATHS

    spark = get_spark_session("Elite_XGBoost_Churn_Model")

    try:
        train_churn_model(spark, PATHS["processed_data"])
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()
