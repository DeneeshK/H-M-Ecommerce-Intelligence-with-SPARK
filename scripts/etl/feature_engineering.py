"""
Feature Engineering Module
=========================
Creates ML-ready features from raw H&M e-commerce data.

Feature Categories:
1. RFM (Recency, Frequency, Monetary)
2. Behavioral (purchase patterns, preferences)
3. Product (category encodings, popularity)
4. Temporal (seasonality, trends)


"""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg as spark_avg,
    min as spark_min, max as spark_max, stddev,
    datediff, lit, when, desc, asc,
    year, month, dayofweek, weekofyear,
    lag, lead, first, last,
    collect_list, collect_set,
    row_number, rank, dense_rank,
    expr, concat_ws
)
from pyspark.sql.types import IntegerType, DoubleType, StringType
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerFeatureEngineer:
    """
    Generates customer-level features from transaction data.
    
    Features Generated:
    - RFM scores (Recency, Frequency, Monetary)
    - Purchase patterns (velocity, consistency)
    - Product preferences (categories, diversity)
    - Temporal patterns (time of day, day of week)
    - Engagement metrics (channel preference, basket size)
    """
    
    def __init__(self, spark, reference_date="2020-09-15"):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        spark : SparkSession
            Active Spark session
        reference_date : str
            Cutoff date for feature calculation (end of training period)
        """
        self.spark = spark
        self.reference_date = reference_date
        logger.info(f"CustomerFeatureEngineer initialized with reference_date={reference_date}")
    
    
    def calculate_rfm_features(self, transactions_df):
        """
        Calculate RFM (Recency, Frequency, Monetary) features.
        
        WHY RFM?
        - Recency: Recent buyers more likely to buy again
        - Frequency: Loyal customers = repeat purchases
        - Monetary: High spenders = high value
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Raw transactions with columns: customer_id, t_dat, price
            
        Returns:
        --------
        DataFrame : Customer-level RFM features
        """
        logger.info("Calculating RFM features...")
        
        rfm_features = transactions_df.groupBy("customer_id").agg(
            
            # RECENCY FEATURES
           
            spark_max("t_dat").alias("last_purchase_date"),
            spark_min("t_dat").alias("first_purchase_date"),
            
            
            # FREQUENCY FEATURES
            
            count("*").alias("total_purchases"),
            countDistinct("t_dat").alias("unique_shopping_days"),
            countDistinct("article_id").alias("unique_products_bought"),
            
           
            # MONETARY FEATURES
           
            spark_sum("price").alias("total_spend"),
            spark_avg("price").alias("avg_order_value"),
            spark_max("price").alias("max_single_purchase"),
            spark_min("price").alias("min_single_purchase"),
            stddev("price").alias("price_std_dev")
        )
        
        # Calculate derived features
        rfm_features = rfm_features.withColumn(
            "days_since_last_purchase",
            datediff(lit(self.reference_date), col("last_purchase_date"))
        ).withColumn(
            "days_since_first_purchase",
            datediff(lit(self.reference_date), col("first_purchase_date"))
        ).withColumn(
            "customer_tenure_days",
            datediff(col("last_purchase_date"), col("first_purchase_date"))
        ).withColumn(
            "purchases_per_day",
            col("total_purchases") / (col("customer_tenure_days") + 1)  # +1 to avoid division by zero
        ).withColumn(
            "spend_per_purchase",
            col("total_spend") / col("total_purchases")
        ).withColumn(
            "products_per_purchase",
            col("unique_products_bought") / col("total_purchases")
        )
        
        logger.info(f" RFM features calculated for {rfm_features.count():,} customers")
        
        return rfm_features
    
    
    def calculate_behavioral_features(self, transactions_df, articles_df):
        """
        Calculate behavioral features (preferences, patterns).
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Transactions with customer_id, article_id, t_dat, sales_channel_id
        articles_df : DataFrame
            Product catalog with article_id, product_group_name, colour_group_name
            
        Returns:
        --------
        DataFrame : Customer behavioral features
        """
        logger.info("Calculating behavioral features...")
        
        # Join transactions with product info
        trans_with_products = transactions_df.join(
            articles_df.select(
                "article_id",
                "product_group_name",
                "colour_group_name",
                "garment_group_name"
            ),
            on="article_id",
            how="left"
        )
        
        
        # CATEGORY PREFERENCES
        
        # Window to rank categories by purchase count per customer
        category_window = Window.partitionBy("customer_id").orderBy(desc("category_count"))
        
        preferred_category = trans_with_products.groupBy(
            "customer_id", "product_group_name"
        ).agg(
            count("*").alias("category_count")
        ).withColumn(
            "category_rank",
            row_number().over(category_window)
        ).filter(
            col("category_rank") == 1
        ).select(
            "customer_id",
            col("product_group_name").alias("preferred_category"),
            col("category_count").alias("preferred_category_count")
        )
        

        # COLOR PREFERENCES
     
        color_window = Window.partitionBy("customer_id").orderBy(desc("color_count"))
        
        preferred_color = trans_with_products.groupBy(
            "customer_id", "colour_group_name"
        ).agg(
            count("*").alias("color_count")
        ).withColumn(
            "color_rank",
            row_number().over(color_window)
        ).filter(
            col("color_rank") == 1
        ).select(
            "customer_id",
            col("colour_group_name").alias("preferred_color"),
            col("color_count").alias("preferred_color_count")
        )
        
        
        # CHANNEL PREFERENCES
        
        channel_prefs = transactions_df.groupBy("customer_id", "sales_channel_id").agg(
            count("*").alias("channel_purchases")
        ).groupBy("customer_id").pivot("sales_channel_id").agg(
            first("channel_purchases")
        ).fillna(0)
        
        # Rename pivot columns
        channel_prefs = channel_prefs.withColumnRenamed("1", "store_purchases") \
                                     .withColumnRenamed("2", "online_purchases")
        
        # Calculate channel preference ratio
        channel_prefs = channel_prefs.withColumn(
            "total_channel_purchases",
            col("store_purchases") + col("online_purchases")
        ).withColumn(
            "online_preference_ratio",
            col("online_purchases") / col("total_channel_purchases")
        )
        
       
        # DIVERSITY METRICS
       
        diversity = trans_with_products.groupBy("customer_id").agg(
            countDistinct("product_group_name").alias("unique_categories"),
            countDistinct("colour_group_name").alias("unique_colors"),
            countDistinct("garment_group_name").alias("unique_garment_groups")
        )
        
        # Join all behavioral features
        behavioral = preferred_category \
            .join(preferred_color, on="customer_id", how="left") \
            .join(channel_prefs, on="customer_id", how="left") \
            .join(diversity, on="customer_id", how="left")
        
        logger.info(f" Behavioral features calculated for {behavioral.count():,} customers")
        
        return behavioral
    
    
    def calculate_temporal_features(self, transactions_df):
        """
        Calculate time-based features (seasonality, patterns).
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Transactions with customer_id, t_dat
            
        Returns:
        --------
        DataFrame : Customer temporal features
        """
        logger.info("Calculating temporal features...")
        
        # Add temporal columns
        trans_temporal = transactions_df.withColumn(
            "month", month("t_dat")
        ).withColumn(
            "day_of_week", dayofweek("t_dat")
        ).withColumn(
            "is_weekend", when(dayofweek("t_dat").isin([1, 7]), 1).otherwise(0)
        )
        
       
        # TEMPORAL AGGREGATIONS
        
        temporal_features = trans_temporal.groupBy("customer_id").agg(
            # Weekend shopping behavior
            spark_sum("is_weekend").alias("weekend_purchases"),
            count("*").alias("total_purchases_for_ratio"),
            
            # Peak month
            spark_max("month").alias("last_purchase_month"),
            
            # Shopping frequency patterns
            countDistinct("t_dat").alias("unique_shopping_days_temporal")
        )
        
        # Calculate ratios
        temporal_features = temporal_features.withColumn(
            "weekend_purchase_ratio",
            col("weekend_purchases") / col("total_purchases_for_ratio")
        ).drop("total_purchases_for_ratio", "unique_shopping_days_temporal")
        
        logger.info(f" Temporal features calculated for {temporal_features.count():,} customers")
        
        return temporal_features
    
    def calculate_purchase_interval_features(self, transactions_df):
        """
        Calculate features based on time between purchases.
        Critical for next-purchase timing prediction.
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Transactions with customer_id, t_dat
            
        Returns:
        --------
        DataFrame : Purchase interval features
        """
        logger.info("Calculating purchase interval features...")
        
        # Window for calculating intervals
        customer_window = Window.partitionBy("customer_id").orderBy("t_dat")
        
        # Calculate days between consecutive purchases
        intervals = transactions_df.select(
            "customer_id",
            "t_dat"
        ).distinct().withColumn(
            "prev_purchase_date",
            lag("t_dat", 1).over(customer_window)
        ).withColumn(
            "days_between_purchases",
            datediff(col("t_dat"), col("prev_purchase_date"))
        ).filter(
            col("days_between_purchases").isNotNull()
        ).filter(
            col("days_between_purchases") > 0
        )
        
        # Aggregate interval statistics per customer
        interval_features = intervals.groupBy("customer_id").agg(
            spark_avg("days_between_purchases").alias("avg_days_between_purchases"),
            spark_min("days_between_purchases").alias("min_days_between_purchases"),
            spark_max("days_between_purchases").alias("max_days_between_purchases"),
            stddev("days_between_purchases").alias("std_days_between_purchases")
        )
        
        logger.info(f" Interval features calculated for {interval_features.count():,} customers")
        
        return interval_features
    
    
    def create_churn_labels(self, transactions_df, churn_threshold=90):
        """
        Create churn labels for churn prediction model.
        
        WHY 90 days?
        - From EDA: Median purchase interval insights
        - Business definition: 90 days = inactive customer
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Transactions with customer_id, t_dat
        churn_threshold : int
            Days without purchase = churned (default 90)
            
        Returns:
        --------
        DataFrame : Churn labels per customer
        """
        logger.info(f"Creating churn labels (threshold={churn_threshold} days)...")
        
        # Get last purchase date per customer
        last_purchases = transactions_df.groupBy("customer_id").agg(
            spark_max("t_dat").alias("last_purchase_date")
        )
        
        # Calculate churn label
        churn_labels = last_purchases.withColumn(
            "days_since_last",
            datediff(lit(self.reference_date), col("last_purchase_date"))
        ).withColumn(
            "is_churned",
            when(col("days_since_last") > churn_threshold, 1).otherwise(0)
        ).withColumn(
            "churn_risk_level",
            when(col("days_since_last") <= 30, "Active")
            .when(col("days_since_last") <= 60, "Moderate Risk")
            .when(col("days_since_last") <= 90, "High Risk")
            .otherwise("Churned")
        ).select(
            "customer_id",
            "is_churned",
            "churn_risk_level"
        )
        
        # Log churn distribution
        churn_dist = churn_labels.groupBy("is_churned").count().collect()
        for row in churn_dist:
            status = "Churned" if row['is_churned'] == 1 else "Active"
            logger.info(f"   {status}: {row['count']:,} customers")
        
        return churn_labels
    
    
    def add_demographic_features(self, customer_features, customers_df):
        """
        Join customer demographics with engineered features.
        
        Parameters:
        -----------
        customer_features : DataFrame
            Engineered customer features
        customers_df : DataFrame
            Raw customer demographics
            
        Returns:
        --------
        DataFrame : Features + demographics
        """
        logger.info("Adding demographic features...")
        
        # Clean demographics - drop useless columns
        demographics_clean = customers_df.select(
            "customer_id",
            "age",
            "club_member_status",
            "fashion_news_frequency"
            # NOTE: FN and Active columns dropped (65%+ null)
        )
        
        # Join with features
        enriched = customer_features.join(
            demographics_clean,
            on="customer_id",
            how="left"
        )
        
        logger.info(f" Demographics added")
        
        return enriched
    
    
    def create_all_customer_features(self, transactions_df, articles_df, customers_df):
        """
        Create complete customer feature set with ALL features.
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Raw transactions
        articles_df : DataFrame
            Product catalog
        customers_df : DataFrame
            Customer demographics
            
        Returns:
        --------
        DataFrame : Complete customer features
        """
        logger.info("="*80)
        logger.info("CREATING COMPLETE CUSTOMER FEATURE SET")
        logger.info("="*80)
        
        # Calculate each feature group
        logger.info("\n1/6: RFM Features...")
        rfm = self.calculate_rfm_features(transactions_df)
        
        logger.info("\n2/6: Behavioral Features...")
        behavioral = self.calculate_behavioral_features(transactions_df, articles_df)
        
        logger.info("\n3/6: Temporal Features...")
        temporal = self.calculate_temporal_features(transactions_df)
        
        logger.info("\n4/6: Purchase Interval Features...")
        intervals = self.calculate_purchase_interval_features(transactions_df)
        
        logger.info("\n5/6: Churn Labels...")
        churn = self.create_churn_labels(transactions_df, churn_threshold=90)
        
        # Join all features
        customer_features = rfm \
            .join(behavioral, on="customer_id", how="left") \
            .join(temporal, on="customer_id", how="left") \
            .join(intervals, on="customer_id", how="left") \
            .join(churn, on="customer_id", how="left")
        
        logger.info("\n6/6: Adding Demographics...")
        customer_features = self.add_demographic_features(customer_features, customers_df)
        
        logger.info(f"\n" + "="*80)
        logger.info(f" COMPLETE FEATURE SET CREATED!")
        logger.info(f"="*80)
        logger.info(f"   Total customers  : {customer_features.count():,}")
        logger.info(f"   Total features   : {len(customer_features.columns)}")
        logger.info(f"   Feature groups   : RFM + Behavioral + Temporal + Intervals + Churn + Demographics")
        logger.info(f"="*80)
        
        return customer_features



# MAIN EXECUTION (for testing)

def main():
    """Test the feature engineering pipeline."""
    from spark_config import get_spark_session, stop_spark_session, PATHS
    from extract import DataExtractor
    
    # Initialize Spark
    spark = get_spark_session("Feature_Engineering_Test")
    
    try:
        # Load data
        print("\n Loading data...")
        extractor = DataExtractor(spark, PATHS["raw_data"])
        articles, customers, transactions = extractor.load_all()
        
        # Initialize feature engineer
        feature_engineer = CustomerFeatureEngineer(
            spark=spark,
            reference_date="2020-09-15"
        )
        
        # Create features
        customer_features = feature_engineer.create_all_customer_features(
            transactions_df=transactions,
            articles_df=articles,
            customers_df=customers
        )
        
        # Show sample
        print("\n" + "="*80)
        print(" SAMPLE CUSTOMER FEATURES")
        print("="*80)
        customer_features.select(
            "customer_id",
            "days_since_last_purchase",
            "total_purchases",
            "total_spend",
            "preferred_category",
            "online_preference_ratio"
        ).show(10, truncate=False)
        
        # Feature statistics
        print("\n" + "="*80)
        print(" FEATURE STATISTICS")
        print("="*80)
        customer_features.select(
            "total_purchases",
            "total_spend",
            "days_since_last_purchase",
            "unique_products_bought"
        ).describe().show()
        
        print("\n Feature engineering test complete!")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        raise
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()