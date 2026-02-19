"""
Elite Customer Churn Prediction Model with Hyperparameter Tuning
================================================================
Includes:
- Time-safe features
- Categorical encoding
- Gradient Boosted Trees
- Hyperparameter tuning using Cross Validation
- Full metrics

Author: Data Engineering Portfolio Project
Date: February 2026
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_churn_model(spark, data_path):
    
    logger.info("="*100)
    logger.info(" ELITE CUSTOMER CHURN MODEL — WITH HYPERPARAMETER TUNING")
    logger.info("="*100)

    # Load data
    train_df = spark.read.parquet(f"{data_path}/churn_train_v1")
    test_df  = spark.read.parquet(f"{data_path}/churn_test_v1")

    logger.info(f" Train customers: {train_df.count():,}")
    logger.info(f" Test customers:  {test_df.count():,}")

    # Numeric + categorical features
    numeric_features = [
        "total_purchases", "total_spend", "avg_order_value",
        "unique_products", "customer_tenure_days", "purchase_frequency",
        "days_since_last_purchase", "avg_days_between_purchases",
        "std_days_between_purchases", "online_ratio", "weekend_ratio"
    ]

    categorical_features = ["club_member_status", "fashion_news_frequency"]

    # String Indexing
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

    gbt = GBTClassifier(
        labelCol="is_churned",
        featuresCol="features",
        seed=42
    )

    pipeline = Pipeline(stages=indexers + [encoder, assembler, gbt])

    # Hyperparameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [4, 6]) \
        .addGrid(gbt.maxIter, [30, 50]) \
        .addGrid(gbt.stepSize, [0.05, 0.1]) \
        .addGrid(gbt.subsamplingRate, [0.8, 1.0]) \
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
        parallelism=4,
        seed=42
    )

    logger.info("\n Training model with hyperparameter tuning (this will take time)...")

    cv_model = crossval.fit(train_df)

    logger.info(" Hyperparameter tuning completed!")

    predictions = cv_model.transform(test_df)

    # Metrics
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
    logger.info(" FINAL TUNED MODEL PERFORMANCE")
    logger.info("="*100)
    logger.info(f"   AUC-ROC:   {auc:.4f}")
    logger.info(f"   Accuracy:  {acc:.4f}")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall:    {recall:.4f}")
    logger.info("="*100)

    # Best model params
    best_gbt = cv_model.bestModel.stages[-1]
    logger.info("\n BEST MODEL PARAMETERS:")
    logger.info(f"   maxDepth: {best_gbt.getMaxDepth()}")
    logger.info(f"   maxIter: {best_gbt.getMaxIter()}")
    logger.info(f"   stepSize: {best_gbt.getStepSize()}")
    logger.info(f"   subsamplingRate: {best_gbt.getSubsamplingRate()}")

    return cv_model


def main():
    import sys, os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    from scripts.etl.spark_config import get_spark_session, stop_spark_session, PATHS

    spark = get_spark_session("Elite_Churn_Model_Tuning")

    try:
        train_churn_model(spark, PATHS["processed_data"])
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()
