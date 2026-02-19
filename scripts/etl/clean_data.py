"""
Data Cleaning Module
===================
Handles missing values, outliers, and data quality issues.

Cleaning Tasks:
1. Drop useless columns (FN, Active - 65%+ null)
2. Impute missing values intelligently
3. Handle outliers
4. Create clean, ML-ready datasets


"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, isnan, isnull,
    mean as spark_mean, median, stddev,
    countDistinct, expr
)
from pyspark.sql.window import Window
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleans and prepares data for ML models.
    """
    
    def __init__(self, spark):
        self.spark = spark
        logger.info("DataCleaner initialized")
    
    
    def clean_customer_features(self, customer_features_df):
        """
        Clean customer features for ML readiness.
        
        Cleaning Steps:
        1. Handle null values in demographics
        2. Cap extreme outliers
        3. Fill nulls in interval features (one-time buyers)
        
        Parameters:
        -----------
        customer_features_df : DataFrame
            Raw customer features with nulls
            
        Returns:
        --------
        DataFrame : Clean customer features
        """
        logger.info("="*80)
        logger.info("CLEANING CUSTOMER FEATURES")
        logger.info("="*80)
        
        df = customer_features_df
        
       
        # 1. IMPUTE MISSING AGE
      
        logger.info("\n1. Imputing missing age values...")
        
        # Calculate median age by club_member_status
        median_ages = df.filter(col("age").isNotNull()) \
            .groupBy("club_member_status") \
            .agg(median("age").alias("median_age"))
        
        # Join and fill nulls
        df = df.join(median_ages, on="club_member_status", how="left")
        
        # First pass: impute based on club status
        df = df.withColumn(
            "age",
            when(col("age").isNull(), col("median_age")).otherwise(col("age"))
        ).drop("median_age")
        
        # Second pass: if still null, use global median (for customers with null club_status)
        global_median = df.filter(col("age").isNotNull()).approxQuantile("age", [0.5], 0.01)[0]
        df = df.withColumn(
            "age",
            when(col("age").isNull(), lit(global_median)).otherwise(col("age"))
        )
        
        age_nulls_after = df.filter(col("age").isNull()).count()
        logger.info(f"    Age nulls remaining: {age_nulls_after:,}")
        
        
        # 2. IMPUTE MISSING CLUB_MEMBER_STATUS
       
        logger.info("\n2. Imputing missing club_member_status...")
        
        df = df.withColumn(
            "club_member_status",
            when(col("club_member_status").isNull(), "ACTIVE")
            .otherwise(col("club_member_status"))
        )
        
        status_nulls_after = df.filter(col("club_member_status").isNull()).count()
        logger.info(f"    Club status nulls remaining: {status_nulls_after:,}")
        
        
        # 3. IMPUTE MISSING FASHION_NEWS_FREQUENCY
        
        logger.info("\n3. Imputing missing fashion_news_frequency...")
        
        df = df.withColumn(
            "fashion_news_frequency",
            when(col("fashion_news_frequency").isNull(), "NONE")
            .otherwise(col("fashion_news_frequency"))
        )
        
        news_nulls_after = df.filter(col("fashion_news_frequency").isNull()).count()
        logger.info(f"    Fashion news nulls remaining: {news_nulls_after:,}")
        
        
        # 4. HANDLE PURCHASE INTERVAL NULLS (One-Time Buyers)
        
        logger.info("\n4. Handling purchase interval nulls (one-time buyers)...")
        
        interval_cols = [
            "avg_days_between_purchases",
            "min_days_between_purchases",
            "max_days_between_purchases",
            "std_days_between_purchases"
        ]
        
        for col_name in interval_cols:
            if col_name in df.columns:
                # Fill with -1 to indicate "not applicable"
                df = df.withColumn(
                    col_name,
                    when(col(col_name).isNull(), -1.0).otherwise(col(col_name))
                )
        
        logger.info(f"    Interval nulls filled with -1 (one-time buyer indicator)")
        
        
        # 5. CAP EXTREME AGE OUTLIERS
        
        logger.info("\n5. Capping extreme age outliers...")
        
        # Cap age at 100 (if any > 100)
        extreme_ages = df.filter(col("age") > 100).count()
        if extreme_ages > 0:
            logger.info(f"   Found {extreme_ages:,} ages > 100, capping at 100")
            df = df.withColumn(
                "age",
                when(col("age") > 100, 100).otherwise(col("age"))
            )
        else:
            logger.info(f"    No extreme ages found")
        
        # 6. FINAL VALIDATION
        
        logger.info("\n6. Final data quality check...")
        
        total_rows = df.count()
        critical_nulls = 0
        
        critical_cols = [
            "customer_id", "total_purchases", "total_spend",
            "days_since_last_purchase", "age", "club_member_status"
        ]
        
        for col_name in critical_cols:
            null_count = df.filter(col(col_name).isNull()).count()
            if null_count > 0:
                logger.warning(f"    {col_name}: {null_count:,} nulls remaining")
                critical_nulls += null_count
        
        if critical_nulls == 0:
            logger.info(f"    No critical nulls remaining!")
        
        logger.info(f"\n{'='*80}")
        logger.info(f" CLEANING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"   Total customers: {total_rows:,}")
        logger.info(f"   Features: {len(df.columns)}")
        logger.info(f"   Data quality: Ready for ML")
        logger.info(f"{'='*80}\n")
        
        return df
    
    
    def create_train_test_split(self, customer_features_df, test_date="2020-09-15"):
        """
        Split data into train/test based on last purchase date.
        
        WHY time-based split?
        - Prevents data leakage
        - Mimics production scenario
        - Proper temporal validation
        
        Parameters:
        -----------
        customer_features_df : DataFrame
            Clean customer features
        test_date : str
            Cutoff date for test set
            
        Returns:
        --------
        tuple : (train_df, test_df)
        """
        logger.info("Creating train/test split...")
        
        # Training: customers with last purchase BEFORE test_date
        train_df = customer_features_df.filter(
            col("last_purchase_date") < test_date
        )
        
        # Test: customers with last purchase ON or AFTER test_date
        test_df = customer_features_df.filter(
            col("last_purchase_date") >= test_date
        )
        
        train_count = train_df.count()
        test_count = test_df.count()
        total = train_count + test_count
        
        logger.info(f"   Train set: {train_count:,} ({train_count/total*100:.1f}%)")
        logger.info(f"   Test set:  {test_count:,} ({test_count/total*100:.1f}%)")
        
        return train_df, test_df


def main():
    """Test data cleaning."""
    from spark_config import get_spark_session, stop_spark_session, PATHS
    
    spark = get_spark_session("Data_Cleaning_Test")
    
    try:
        # Load engineered features
        print("\n Loading customer features from feature store...")
        customer_features = spark.read.parquet(
            f"{PATHS['processed_data']}/customer_features_v1"
        )
        
        print(f" Loaded {customer_features.count():,} customers with {len(customer_features.columns)} features")
        
        # Clean data
        cleaner = DataCleaner(spark)
        clean_features = cleaner.clean_customer_features(customer_features)
        
        # Create train/test split
        train, test = cleaner.create_train_test_split(clean_features)
        
        # Show sample
        print("\n" + "="*80)
        print(" SAMPLE CLEAN FEATURES")
        print("="*80)
        clean_features.select(
            "customer_id",
            "total_purchases",
            "days_since_last_purchase",
            "age",
            "club_member_status",
            "is_churned"
        ).show(10, truncate=False)
        
        print("\n Data cleaning test complete!")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        raise
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()