"""
Recommendation Feature Engineering - Temporal-Safe
=================================================
Features for product recommendation system.

CRITICAL: No data leakage - uses ONLY training period data.

Author: Data Engineering Portfolio Project
Date: February 2026
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg as spark_avg,
    row_number, desc, collect_list, collect_set, concat_ws,
    explode, array, struct, lit, when, size
)
from pyspark.ml.fpm import FPGrowth
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationFeatureEngineer:
    """
    Feature engineering for recommendation system.
    """
    
    def __init__(self, spark, train_end_date="2020-06-24"):
        """
        Initialize with temporal boundary.
        
        Parameters:
        -----------
        spark : SparkSession
        train_end_date : str
            Cutoff date for training (2020-06-24)
        """
        self.spark = spark
        self.train_end_date = train_end_date
        self.test_start_date = train_end_date
        self.test_end_date = "2020-09-22"
        
        logger.info("="*80)
        logger.info("RecommendationFeatureEngineer initialized")
        logger.info("="*80)
        logger.info(f"   Training Period:  2018-09-20 to {train_end_date}")
        logger.info(f"   Test Period:      {self.test_start_date} to {self.test_end_date}")
        logger.info("="*80)
    
    
    def build_popularity_model(self, transactions_df, articles_df, top_k=100):
        """
        Build popularity-based recommendations.
        
        WHY this works:
        - Popular items are safe bets
        - Handles cold-start
        - Fast to compute
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Training transactions ONLY
        articles_df : DataFrame
            Product catalog
        top_k : int
            Number of popular items to track
            
        Returns:
        --------
        DataFrame : Popular items with scores
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL 1: POPULARITY BASELINE")
        logger.info("="*80)
        
        # Count purchases per article
        popularity = transactions_df.groupBy("article_id").agg(
            count("*").alias("purchase_count"),
            countDistinct("customer_id").alias("unique_customers")
        ).withColumn(
            "popularity_score",
            col("purchase_count") * 0.7 + col("unique_customers") * 0.3
        ).orderBy(desc("popularity_score"))
        
        # Get top K
        top_popular = popularity.limit(top_k)
        
        # Add product info
        popular_with_info = top_popular.join(
            articles_df.select(
                "article_id",
                "prod_name",
                "product_group_name",
                "colour_group_name"
            ),
            on="article_id",
            how="left"
        )
        
        logger.info(f"✅ Top {top_k} popular items identified")
        logger.info(f"   Most popular: {popular_with_info.first()['article_id']}")
        
        return popular_with_info
    
    
    def build_customer_preference_features(self, transactions_df, articles_df):
        """
        Build customer preference profiles.
        
        For each customer, learn:
        - Favorite categories
        - Favorite colors
        - Price range
        - Purchase recency
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Training transactions
        articles_df : DataFrame
            Product catalog
            
        Returns:
        --------
        DataFrame : Customer preferences
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL 2: CUSTOMER PREFERENCE PROFILES")
        logger.info("="*80)
        
        # Join with product attributes
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
        
        # ══════════════════════════════════════════════════════════
        # CUSTOMER PURCHASE HISTORY
        # ══════════════════════════════════════════════════════════
        customer_history = trans_with_products.groupBy("customer_id").agg(
            collect_set("article_id").alias("purchased_items"),
            collect_list("product_group_name").alias("all_categories"),
            collect_list("colour_group_name").alias("all_colors"),
            count("*").alias("total_purchases"),
            spark_avg("price").alias("avg_price")
        )
        
        # ══════════════════════════════════════════════════════════
        # FAVORITE CATEGORY (most frequent)
        # ══════════════════════════════════════════════════════════
        category_window = Window.partitionBy("customer_id").orderBy(desc("category_count"))
        
        favorite_category = trans_with_products.groupBy(
            "customer_id", "product_group_name"
        ).agg(
            count("*").alias("category_count")
        ).withColumn(
            "rank",
            row_number().over(category_window)
        ).filter(col("rank") == 1).select(
            "customer_id",
            col("product_group_name").alias("favorite_category"),
            col("category_count").alias("favorite_category_count")
        )
        
        # ══════════════════════════════════════════════════════════
        # FAVORITE COLOR
        # ══════════════════════════════════════════════════════════
        color_window = Window.partitionBy("customer_id").orderBy(desc("color_count"))
        
        favorite_color = trans_with_products.groupBy(
            "customer_id", "colour_group_name"
        ).agg(
            count("*").alias("color_count")
        ).withColumn(
            "rank",
            row_number().over(color_window)
        ).filter(col("rank") == 1).select(
            "customer_id",
            col("colour_group_name").alias("favorite_color"),
            col("color_count").alias("favorite_color_count")
        )
        
        # ══════════════════════════════════════════════════════════
        # COMBINE PREFERENCES
        # ══════════════════════════════════════════════════════════
        preferences = customer_history \
            .join(favorite_category, on="customer_id", how="left") \
            .join(favorite_color, on="customer_id", how="left")
        
        logger.info(f"✅ Preference profiles for {preferences.count():,} customers")
        
        return preferences
    
    
    def build_copurchase_rules(self, transactions_df, min_support=0.001, min_confidence=0.1):
        """
        Build association rules using FP-Growth.
        
        Discovers patterns like:
        "Customers who bought A also bought B"
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Training transactions
        min_support : float
            Minimum frequency (0.001 = 0.1% of transactions)
        min_confidence : float
            Minimum rule confidence
            
        Returns:
        --------
        DataFrame : Association rules
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL 3: CO-PURCHASE ASSOCIATION RULES")
        logger.info("="*80)
        logger.info(f"   Min Support: {min_support}")
        logger.info(f"   Min Confidence: {min_confidence}")
        
        # ══════════════════════════════════════════════════════════
        # CREATE BASKETS (transactions per customer per day)
        # ══════════════════════════════════════════════════════════
        logger.info("\n📦 Creating shopping baskets...")
        
        baskets = transactions_df.groupBy("customer_id", "t_dat").agg(
            collect_set("article_id").alias("items")
        ).select("items")
        
        # Filter baskets with at least 2 items
        baskets = baskets.filter(size(col("items")) >= 2)
        
        basket_count = baskets.count()
        logger.info(f"   ✅ {basket_count:,} baskets with 2+ items")
        
        # ══════════════════════════════════════════════════════════
        # RUN FP-GROWTH
        # ══════════════════════════════════════════════════════════
        logger.info("\n🔍 Mining frequent itemsets (this may take 3-5 minutes)...")
        
        fp_growth = FPGrowth(
            itemsCol="items",
            minSupport=min_support,
            minConfidence=min_confidence
        )
        
        model = fp_growth.fit(baskets)
        
        # Get association rules
        rules = model.associationRules
        
        # Filter to single-item antecedents (simpler rules)
        rules_filtered = rules.filter(size(col("antecedent")) == 1)
        
        # Extract items from arrays
        rules_final = rules_filtered.withColumn(
            "item_bought",
            col("antecedent")[0]
        ).withColumn(
            "item_recommended",
            col("consequent")[0]
        ).select(
            "item_bought",
            "item_recommended",
            col("confidence").alias("copurchase_score")
        ).orderBy(desc("copurchase_score"))
        
        rule_count = rules_final.count()
        logger.info(f"✅ Discovered {rule_count:,} co-purchase rules")
        
        if rule_count > 0:
            logger.info("\n📊 Sample rules:")
            rules_final.show(5, truncate=False)
        
        return rules_final
    
    
    def create_ground_truth_labels(self, transactions_df):
        """
        Create ground truth: what customers ACTUALLY bought in test period.
        
        This is used for evaluation (MAP@12).
        
        Parameters:
        -----------
        transactions_df : DataFrame
            ALL transactions
            
        Returns:
        --------
        DataFrame : Customer → actual purchased items in test period
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING GROUND TRUTH LABELS")
        logger.info("="*80)
        
        # Test period transactions
        test_transactions = transactions_df.filter(
            (col("t_dat") >= self.test_start_date) &
            (col("t_dat") <= self.test_end_date)
        )
        
        # Group by customer
        ground_truth = test_transactions.groupBy("customer_id").agg(
            collect_set("article_id").alias("actual_purchased")
        )
        
        logger.info(f"✅ Ground truth for {ground_truth.count():,} customers")
        
        # Show distribution
        avg_items = test_transactions.groupBy("customer_id").agg(
            countDistinct("article_id").alias("item_count")
        )
        
        stats = avg_items.select(
            spark_avg("item_count").alias("avg"),
            spark_sum(when(col("item_count") == 1, 1).otherwise(0)).alias("one_item")
        ).collect()[0]
        
        logger.info(f"   Avg items per customer: {stats['avg']:.2f}")
        logger.info(f"   Customers with 1 item: {stats['one_item']:,}")
        
        return ground_truth


def main():
    """Test recommendation feature engineering."""
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from scripts.etl.spark_config import get_spark_session, stop_spark_session, PATHS
    from scripts.etl.extract import DataExtractor
    
    spark = get_spark_session("Recommendation_Features_Test")
    
    try:
        # Load data
        logger.info("\n📥 Loading data...")
        extractor = DataExtractor(spark, PATHS["raw_data"])
        articles, customers, transactions = extractor.load_all()
        
        # Initialize
        rec_engineer = RecommendationFeatureEngineer(
            spark,
            train_end_date="2020-06-24"
        )
        
        # Split data
        train_trans = transactions.filter(col("t_dat") < rec_engineer.train_end_date)
        
        # Build features
        popularity = rec_engineer.build_popularity_model(train_trans, articles, top_k=100)
        
        preferences = rec_engineer.build_customer_preference_features(train_trans, articles)
        
        copurchase = rec_engineer.build_copurchase_rules(train_trans)
        
        ground_truth = rec_engineer.create_ground_truth_labels(transactions)
        
        logger.info("\n" + "="*80)
        logger.info("✅ RECOMMENDATION FEATURES COMPLETE")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()