"""
Data Cleaning Module - Temporal-Safe Version
============================================
Handles missing values and data quality for churn model.

Author: Data Engineering Portfolio Project
Date: February 2026 (Fixed)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, median, stddev
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleans customer features for ML readiness.
    """
    
    def __init__(self, spark):
        self.spark = spark
        logger.info("DataCleaner initialized")
    
    
    def clean_churn_features(self, features_df):
        """
        Clean features for churn model.
        
        Cleaning:
        1. Impute missing age (median by club status, then global)
        2. Impute missing club_member_status (mode = ACTIVE)
        3. Impute missing fashion_news_frequency (mode = NONE)
        4. Fill nulls in interval features (one-time buyers already handled)
        5. Drop rows with critical nulls
        
        Parameters:
        -----------
        features_df : DataFrame
            Raw features with potential nulls
            
        Returns:
        --------
        DataFrame : Clean features ready for ML
        """
        logger.info("="*80)
        logger.info("CLEANING FEATURES FOR CHURN MODEL")
        logger.info("="*80)
        
        df = features_df
        initial_count = df.count()
        
        
        # 1. IMPUTE AGE
    
        logger.info("\n1 Imputing missing age...")
        
        age_null_before = df.filter(col("age").isNull()).count()
        logger.info(f"   Age nulls before: {age_null_before:,}")
        
        # Median by club status
        median_ages = df.filter(col("age").isNotNull()) \
            .groupBy("club_member_status") \
            .agg(median("age").alias("median_age"))
        
        df = df.join(median_ages, on="club_member_status", how="left")
        df = df.withColumn(
            "age",
            when(col("age").isNull(), col("median_age")).otherwise(col("age"))
        ).drop("median_age")
        
        # Global median for remaining nulls
        global_median = df.filter(col("age").isNotNull()) \
            .approxQuantile("age", [0.5], 0.01)[0]
        
        df = df.withColumn(
            "age",
            when(col("age").isNull(), global_median).otherwise(col("age"))
        )
        
        age_null_after = df.filter(col("age").isNull()).count()
        logger.info(f"   Age nulls after: {age_null_after:,}")
        
        
        # 2. IMPUTE CLUB_MEMBER_STATUS
        
        logger.info("\n2 Imputing club_member_status...")
        
        df = df.withColumn(
            "club_member_status",
            when(col("club_member_status").isNull(), "ACTIVE")
            .otherwise(col("club_member_status"))
        )
        
        
        # 3. IMPUTE FASHION_NEWS_FREQUENCY
        
        logger.info("\n3 Imputing fashion_news_frequency...")
        
        df = df.withColumn(
            "fashion_news_frequency",
            when(col("fashion_news_frequency").isNull(), "NONE")
            .otherwise(col("fashion_news_frequency"))
        )
        
       
        # 4. DROP ROWS WITH CRITICAL NULLS
        
        logger.info("\n4 Dropping rows with critical nulls...")
        
        critical_cols = [
            "customer_id",
            "days_since_last_purchase",
            "total_purchases",
            "is_churned"
        ]
        
        for col_name in critical_cols:
            null_count = df.filter(col(col_name).isNull()).count()
            if null_count > 0:
                logger.warning(f"   Dropping {null_count:,} rows with null {col_name}")
                df = df.filter(col(col_name).isNotNull())
        
        final_count = df.count()
        dropped = initial_count - final_count
        
        logger.info(f"\n Cleaning complete:")
        logger.info(f"   Initial rows: {initial_count:,}")
        logger.info(f"   Final rows:   {final_count:,}")
        logger.info(f"   Dropped:      {dropped:,} ({dropped/initial_count*100:.2f}%)")
        
        return df
    
    
    def create_train_val_test_split(self, features_df, train_ratio=0.7, val_ratio=0.15):
        """
        Split data into train/validation/test for model development.
        
        WHY we need 3 splits:
        - Train: Fit model
        - Validation: Tune hyperparameters
        - Test: Final evaluation (never seen during development)
        
        Parameters:
        -----------
        features_df : DataFrame
            Clean features with labels
        train_ratio : float
            Fraction for training (0.7 = 70%)
        val_ratio : float
            Fraction for validation (0.15 = 15%)
            Test gets the remainder (0.15 = 15%)
            
        Returns:
        --------
        tuple : (train_df, val_df, test_df)
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING TRAIN/VALIDATION/TEST SPLIT")
        logger.info("="*80)
        
        # Random split with seed for reproducibility
        train_df, val_df, test_df = features_df.randomSplit(
            [train_ratio, val_ratio, 1 - train_ratio - val_ratio],
            seed=42
        )
        
        train_count = train_df.count()
        val_count = val_df.count()
        test_count = test_df.count()
        total = train_count + val_count + test_count
        
        logger.info(f"   Train:      {train_count:,} ({train_count/total*100:.1f}%)")
        logger.info(f"   Validation: {val_count:,} ({val_count/total*100:.1f}%)")
        logger.info(f"   Test:       {test_count:,} ({test_count/total*100:.1f}%)")
        
        # Check churn distribution in each split
        logger.info("\n   Churn distribution:")
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            churned = df.filter(col("is_churned") == 1).count()
            total_split = df.count()
            logger.info(f"      {name}: {churned/total_split*100:.1f}% churned")
        
        return train_df, val_df, test_df


def main():
    """Test data cleaning."""
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from scripts.etl.spark_config import get_spark_session, stop_spark_session, PATHS
    from scripts.etl.extract import DataExtractor
    from scripts.etl.feature_engineering import ChurnFeatureEngineer
    
    spark = get_spark_session("Clean_Data_Test")
    
    try:
        # Load data
        extractor = DataExtractor(spark, PATHS["raw_data"])
        articles, customers, transactions = extractor.load_all()
        
        # Create features
        engineer = ChurnFeatureEngineer(spark, train_end_date="2020-06-24")
        features = engineer.create_churn_features(transactions, customers)
        
        # Clean
        cleaner = DataCleaner(spark)
        clean_features = cleaner.clean_churn_features(features)
        
        # Split
        train, val, test = cleaner.create_train_val_test_split(clean_features)
        
        print("\n Cleaning test complete!")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()