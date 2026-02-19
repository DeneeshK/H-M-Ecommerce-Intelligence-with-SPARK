"""
Complete Transform Pipeline - Churn Model
=========================================
End-to-end pipeline: Extract → Engineer → Clean → Save

Author: Data Engineering Portfolio Project
Date: February 2026 (Fixed)
"""

from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_churn_pipeline(spark_session_name="Churn_Pipeline", version="v1"):
    """
    Execute complete churn model ETL pipeline.
    
    Steps:
    1. Extract raw data
    2. Engineer features (temporal-safe)
    3. Clean data (impute nulls)
    4. Split train/val/test
    5. Save to Parquet
    
    Parameters:
    -----------
    spark_session_name : str
    version : str
        Feature version tag
    """
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from scripts.etl.spark_config import get_spark_session, stop_spark_session, PATHS
    from scripts.etl.extract import DataExtractor
    from scripts.etl.feature_engineering import ChurnFeatureEngineer
    from scripts.etl.clean_data import DataCleaner
    from scripts.etl.load import FeatureStoreLoader
    
    logger.info("="*80)
    logger.info("🚀 CHURN MODEL - COMPLETE ETL PIPELINE")
    logger.info("="*80)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Version: {version}")
    logger.info("="*80)
    
    spark = get_spark_session(spark_session_name)
    
    try:
        # ══════════════════════════════════════════════════════════
        # STEP 1: EXTRACT
        # ══════════════════════════════════════════════════════════
        logger.info("\n📥 STEP 1/5: EXTRACT RAW DATA")
        logger.info("="*80)
        
        extractor = DataExtractor(spark, PATHS["raw_data"])
        articles, customers, transactions = extractor.load_all()
        
        # ══════════════════════════════════════════════════════════
        # STEP 2: FEATURE ENGINEERING
        # ══════════════════════════════════════════════════════════
        logger.info("\n🛠️  STEP 2/5: ENGINEER FEATURES (TEMPORAL-SAFE)")
        logger.info("="*80)
        
        engineer = ChurnFeatureEngineer(
            spark,
            train_end_date="2020-06-24",  # Safe date for 90-day window
            prediction_window_days=90
        )
        features = engineer.create_churn_features(transactions, customers)
        
        # ══════════════════════════════════════════════════════════
        # STEP 3: CLEAN DATA
        # ══════════════════════════════════════════════════════════
        logger.info("\n🧹 STEP 3/5: CLEAN DATA")
        logger.info("="*80)
        
        cleaner = DataCleaner(spark)
        clean_features = cleaner.clean_churn_features(features)
        
        # ══════════════════════════════════════════════════════════
        # STEP 4: SPLIT DATA
        # ══════════════════════════════════════════════════════════
        logger.info("\n✂️  STEP 4/5: SPLIT TRAIN/VAL/TEST")
        logger.info("="*80)
        
        train, val, test = cleaner.create_train_val_test_split(
            clean_features,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        # ══════════════════════════════════════════════════════════
        # STEP 5: SAVE TO PARQUET
        # ══════════════════════════════════════════════════════════
        logger.info("\n💾 STEP 5/5: SAVE TO FEATURE STORE")
        logger.info("="*80)
        
        loader = FeatureStoreLoader(spark, PATHS["processed_data"])
        paths = loader.save_churn_features(train, val, test, version=version)
        
        # ══════════════════════════════════════════════════════════
        # PIPELINE COMPLETE
        # ══════════════════════════════════════════════════════════
        logger.info("\n" + "="*80)
        logger.info("🎉 CHURN PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"   Train:      {train.count():,} customers")
        logger.info(f"   Validation: {val.count():,} customers")
        logger.info(f"   Test:       {test.count():,} customers")
        logger.info(f"   Features:   {len(clean_features.columns)}")
        logger.info(f"\n📁 OUTPUT:")
        logger.info(f"   {paths['train_path']}")
        logger.info(f"   {paths['val_path']}")
        logger.info(f"   {paths['test_path']}")
        logger.info("="*80)
        
        return paths
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    print("="*80)
    print("RUNNING COMPLETE CHURN PIPELINE")
    print("="*80)
    
    paths = run_churn_pipeline(version="v1")
    
    print("\n" + "="*80)
    print("✅ READY FOR MODEL TRAINING")
    print("="*80)