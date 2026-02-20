"""
Feature Engineering - Temporal-Safe Implementation
==================================================
CRITICAL: No data leakage. Features use ONLY historical data.

For Churn Prediction:
- Features: Calculated from training period ONLY
- Labels: Calculated from prediction period (future)
- Proper time-based split

Author: Data Engineering Portfolio Project
Date: February 2026 (Fixed Version)
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg as spark_avg,
    min as spark_min, max as spark_max, stddev,
    datediff, lit, when, lag,
    year, month, dayofweek
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnFeatureEngineer:
    """
    Feature engineering for churn prediction with NO DATA LEAKAGE.
    """
    
    def __init__(self, spark, train_end_date="2020-06-24", prediction_window_days=90):
        """
        Initialize feature engineer with strict temporal boundaries.
        
        CRITICAL DATE CALCULATION:
        - Dataset ends: 2020-09-22
        - Prediction window: 90 days
        - Safe train_end: 2020-09-22 - 90 days = 2020-06-24
        - This ensures full 90-day window for churn labeling
        
        Parameters:
        -----------
        spark : SparkSession
        train_end_date : str
            Last date to use for feature calculation (2020-06-24)
        prediction_window_days : int
            Days into future to check for churn (90)
        """
        self.spark = spark
        self.train_end_date = train_end_date
        self.prediction_window_days = prediction_window_days
        
        # Calculate prediction end date
        from datetime import datetime, timedelta
        train_end = datetime.strptime(train_end_date, "%Y-%m-%d")
        prediction_end = train_end + timedelta(days=prediction_window_days)
        self.prediction_end_date = prediction_end.strftime("%Y-%m-%d")
        
        logger.info("="*80)
        logger.info("ChurnFeatureEngineer initialized (TEMPORAL-SAFE)")
        logger.info("="*80)
        logger.info(f"   Dataset Range:          2018-09-20 to 2020-09-22 (733 days)")
        logger.info(f"   Training Period End:    {train_end_date}")
        logger.info(f"   Prediction Window:      {prediction_window_days} days")
        logger.info(f"   Prediction Period End:  {self.prediction_end_date}")
        logger.info(f"   Full 90-day window available for churn labeling")
        logger.info("="*80)
    
    
    def create_churn_features(self, transactions_df, customers_df):
        """
        Create features for churn prediction with proper temporal split.
        
        CRITICAL: All features calculated using ONLY training period data.
        
        Parameters:
        -----------
        transactions_df : DataFrame
            ALL transactions (we'll split inside)
        customers_df : DataFrame
            Customer demographics
            
        Returns:
        --------
        DataFrame : Customers with features and churn labels
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING CHURN FEATURES (TEMPORAL-SAFE)")
        logger.info("="*80)
        
        # ══════════════════════════════════════════════════════════
        # STEP 1: SPLIT DATA PROPERLY
        # ══════════════════════════════════════════════════════════
        logger.info("\n1 Splitting data by time...")
        
        # Training transactions (features)
        train_trans = transactions_df.filter(col("t_dat") < self.train_end_date)
        
        # Prediction period transactions (labels only)
        prediction_trans = transactions_df.filter(
            (col("t_dat") >= self.train_end_date) &
            (col("t_dat") < self.prediction_end_date)
        )
        
        logger.info(f"   Training transactions:    {train_trans.count():,}")
        logger.info(f"   Prediction transactions:  {prediction_trans.count():,}")
        
        # ══════════════════════════════════════════════════════════
        # STEP 2: CALCULATE RFM FEATURES (Training period only)
        # ══════════════════════════════════════════════════════════
        logger.info("\n2 Calculating RFM features from training data...")
        
        rfm = train_trans.groupBy("customer_id").agg(
            # Recency
            spark_max("t_dat").alias("last_purchase_date"),
            spark_min("t_dat").alias("first_purchase_date"),
            
            # Frequency
            count("*").alias("total_purchases"),
            countDistinct("t_dat").alias("unique_shopping_days"),
            countDistinct("article_id").alias("unique_products"),
            
            # Monetary
            spark_sum("price").alias("total_spend"),
            spark_avg("price").alias("avg_order_value")
        )
        
        # Calculate derived features
        rfm = rfm.withColumn(
            "days_since_last_purchase",
            datediff(lit(self.train_end_date), col("last_purchase_date"))
        ).withColumn(
            "customer_tenure_days",
            datediff(lit(self.train_end_date), col("first_purchase_date"))
        ).withColumn(
            "purchase_frequency",
            col("total_purchases") / (col("customer_tenure_days") + 1)
        )
        
        logger.info(f"    RFM features for {rfm.count():,} customers")
        
        # ══════════════════════════════════════════════════════════
        # STEP 3: PURCHASE INTERVAL FEATURES (Training only)
        # ══════════════════════════════════════════════════════════
        logger.info("\n3 Calculating purchase intervals...")
        
        # Window for intervals
        customer_window = Window.partitionBy("customer_id").orderBy("t_dat")
        
        intervals = train_trans.select(
            "customer_id", "t_dat"
        ).distinct().withColumn(
            "prev_date",
            lag("t_dat", 1).over(customer_window)
        ).withColumn(
            "days_between",
            datediff(col("t_dat"), col("prev_date"))
        ).filter(
            col("days_between").isNotNull()
        )
        
        interval_features = intervals.groupBy("customer_id").agg(
            spark_avg("days_between").alias("avg_days_between_purchases"),
            stddev("days_between").alias("std_days_between_purchases")
        )
        
        logger.info(f"  Interval features for {interval_features.count():,} customers")
        
        # ══════════════════════════════════════════════════════════
        # STEP 4: BEHAVIORAL FEATURES (Training only)
        # ══════════════════════════════════════════════════════════
        logger.info("\n4 Calculating behavioral features...")
        
        # Channel preference
        channel_pref = train_trans.groupBy("customer_id").agg(
            spark_sum(when(col("sales_channel_id") == 2, 1).otherwise(0)).alias("online_purchases"),
            count("*").alias("total_trans")
        ).withColumn(
            "online_ratio",
            col("online_purchases") / col("total_trans")
        ).select("customer_id", "online_ratio")
        
        # Weekend shopping
        weekend_pref = train_trans.withColumn(
            "is_weekend",
            when(dayofweek("t_dat").isin([1, 7]), 1).otherwise(0)
        ).groupBy("customer_id").agg(
            spark_avg("is_weekend").alias("weekend_ratio")
        )
        
        logger.info(f" Behavioral features calculated")
        
        # ══════════════════════════════════════════════════════════
        # STEP 5: CREATE CHURN LABELS (Prediction period)
        # ══════════════════════════════════════════════════════════
        logger.info("\n5 Creating churn labels from prediction period...")
        
        # Customers who purchased in prediction window = NOT churned
        active_customers = prediction_trans.select("customer_id").distinct() \
            .withColumn("is_churned", lit(0))
        
        # All customers from training
        all_train_customers = train_trans.select("customer_id").distinct()
        
        # Left join to find churned customers
        labels = all_train_customers.join(
            active_customers,
            on="customer_id",
            how="left"
        ).withColumn(
            "is_churned",
            when(col("is_churned").isNull(), 1).otherwise(0)
        )
        
        churn_count = labels.filter(col("is_churned") == 1).count()
        active_count = labels.filter(col("is_churned") == 0).count()
        total = labels.count()
        
        logger.info(f"   Churned:  {churn_count:,} ({churn_count/total*100:.1f}%)")
        logger.info(f"   Active:   {active_count:,} ({active_count/total*100:.1f}%)")
        
        # ══════════════════════════════════════════════════════════
        # STEP 6: JOIN DEMOGRAPHICS
        # ══════════════════════════════════════════════════════════
        logger.info("\n6 Adding demographics...")
        
        demographics = customers_df.select(
            "customer_id",
            "age",
            "club_member_status",
            "fashion_news_frequency"
        )
        
        # ══════════════════════════════════════════════════════════
        # STEP 7: COMBINE ALL FEATURES
        # ══════════════════════════════════════════════════════════
        logger.info("\n 7 Combining all features...")
        
        features = rfm \
            .join(interval_features, on="customer_id", how="left") \
            .join(channel_pref, on="customer_id", how="left") \
            .join(weekend_pref, on="customer_id", how="left") \
            .join(labels, on="customer_id", how="inner") \
            .join(demographics, on="customer_id", how="left")
        
        # Fill nulls for one-time buyers
        features = features.fillna({
            "avg_days_between_purchases": 999,  # No repeat = large value
            "std_days_between_purchases": 0,
            "online_ratio": 0.5,  # Neutral
            "weekend_ratio": 0.3   # Average
        })
        
        logger.info(f"\n Complete feature set: {features.count():,} customers")
        logger.info(f"   Total features: {len(features.columns)}")
        
        return features


def main():
    """Test feature engineering."""
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from scripts.etl.spark_config import get_spark_session, stop_spark_session, PATHS
    from scripts.etl.extract import DataExtractor
    
    spark = get_spark_session("Churn_Features_Test")
    
    try:
        # Load data
        print("\n Loading data...")
        extractor = DataExtractor(spark, PATHS["raw_data"])
        articles, customers, transactions = extractor.load_all()
        
        # Create features
        engineer = ChurnFeatureEngineer(
            spark,
            train_end_date="2020-06-24",
            prediction_window_days=90
        )
        
        features = engineer.create_churn_features(transactions, customers)
        
        # Show sample
        print("\n" + "="*80)
        print("SAMPLE FEATURES")
        print("="*80)
        features.select(
            "customer_id",
            "days_since_last_purchase",
            "total_purchases",
            "avg_days_between_purchases",
            "is_churned"
        ).show(10)
        
        # Show statistics
        print("\n" + "="*80)
        print("CHURN DISTRIBUTION")
        print("="*80)
        features.groupBy("is_churned").count().show()
        
        print("\n Feature engineering test complete!")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()