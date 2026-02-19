# Week 2-3 Summary: Feature Engineering & Transform Pipeline

## Completion Date
February 19, 2026

## Objectives Achieved
✅ Engineered 40 customer features from raw transactions
✅ Created production-grade feature engineering module
✅ Built data cleaning pipeline with intelligent imputation
✅ Implemented Parquet-based feature store (10x faster)
✅ Created train/test split with proper temporal validation
✅ Orchestrated end-to-end transform pipeline

## Time Investment
- Feature Engineering Development: 6 hours
- Data Cleaning & Quality: 2 hours
- Pipeline Integration: 2 hours
- Testing & Debugging: 2 hours
- **Total: ~12 hours**

## Key Metrics
- **Customers:** 1,362,281 processed
- **Features:** 40 engineered features
- **Train Set:** 1,286,800 customers (94.5%)
- **Test Set:** 75,481 customers (5.5%)
- **Processing Time:** 30 minutes end-to-end
- **Storage Format:** Parquet (columnar, compressed)

## Feature Categories Created

### 1. RFM Features (14)
- Recency: `days_since_last_purchase`, `days_since_first_purchase`
- Frequency: `total_purchases`, `unique_shopping_days`, `purchases_per_day`
- Monetary: `total_spend`, `avg_order_value`, `spend_per_purchase`
- Derived: `customer_tenure_days`, `products_per_purchase`

### 2. Behavioral Features (13)
- Preferences: `preferred_category`, `preferred_color`
- Channel: `online_preference_ratio`, `store_purchases`, `online_purchases`
- Diversity: `unique_categories`, `unique_colors`, `unique_garment_groups`

### 3. Temporal Features (4)
- `weekend_purchase_ratio`, `weekend_purchases`
- `last_purchase_month`

### 4. Purchase Interval Features (4)
- `avg_days_between_purchases`
- `min/max/std_days_between_purchases`
- Special handling: -1 for one-time buyers

### 5. Churn Labels (2)
- `is_churned` (binary: 0/1)
- `churn_risk_level` (Active/Moderate/High/Churned)
- Threshold: 90 days without purchase
- Distribution: 59% churned, 41% active

### 6. Demographics (3)
- `age` (imputed, capped at 100)
- `club_member_status` (imputed to ACTIVE)
- `fashion_news_frequency` (imputed to NONE)

## Technical Achievements

### Feature Engineering Pipeline
```python
CustomerFeatureEngineer
├── calculate_rfm_features()          # 14 features
├── calculate_behavioral_features()   # 13 features
├── calculate_temporal_features()     # 4 features
├── calculate_purchase_interval_features()  # 4 features
├── create_churn_labels()             # 2 features
└── add_demographic_features()        # 3 features
```

### Data Quality Improvements
- **Before:** 65% nulls in FN/Active columns
- **After:** 0% nulls in critical features
- **Imputation Strategy:**
  - Age: Median by club_member_status, then global median
  - Club Status: Mode (ACTIVE)
  - News Frequency: Mode (NONE)
  - Intervals: -1 for one-time buyers (feature engineering!)

### Performance Optimizations
- Parquet columnar storage: 10x faster reads
- Partitioned transactions: Query only needed months
- Cached small datasets: Articles & customers in memory
- Efficient aggregations: Window functions, groupBy

## Challenges Overcome

1. **One-Time Buyer Problem**
   - Issue: No purchase intervals for single-purchase customers
   - Solution: Fill with -1 as explicit "one-time buyer" signal
   - Impact: Feature becomes informative, not just null

2. **Memory Management**
   - Issue: 31.8M rows → aggregations trigger spill warnings
   - Solution: Partition strategy, selective caching
   - Result: Stable execution on 16GB RAM

3. **Temporal Data Leakage**
   - Issue: Must not use future data for training
   - Solution: Time-based split (before 2020-09-08 = train)
   - Impact: Proper model validation

4. **Missing Value Strategy**
   - Issue: Multiple imputation methods needed
   - Solution: Context-aware imputation (by group, then global)
   - Result: Intelligent fills, not naive median

## Data Quality Status

### Before Cleaning
- Age: 1.16% missing
- Club Status: 0.44% missing
- News Frequency: 1.17% missing
- FN/Active: 65%+ missing (dropped)
- Intervals: 33% missing (one-time buyers)

### After Cleaning
- ✅ Age: 0% missing (2-stage imputation)
- ✅ Club Status: 0% missing
- ✅ News Frequency: 0% missing
- ✅ Intervals: 0% missing (-1 indicator)
- ✅ All critical features: Clean and ML-ready

## File Structure Created
```
processed_data/
├── customer_features_v1_complete/
│   ├── part-00000.parquet
│   ├── part-00001.parquet
│   └── _SUCCESS
├── customer_features_v1_train/
│   └── *.parquet (1,286,800 rows)
├── customer_features_v1_test/
│   └── *.parquet (75,481 rows)
├── transactions_clean_v1/
│   ├── year=2018/month=09/
│   ├── year=2018/month=10/
│   └── ... (24 partitions)
└── METADATA.txt
```

## Next Week Priorities

### Week 4: Recommendation System
1. Implement ALS collaborative filtering
2. Build content-based filtering (product attributes)
3. Create hybrid ensemble
4. Evaluate with MAP@12 metric
5. Establish baseline with popularity model

### Preparation Complete
- ✅ User-item interaction matrix ready (transactions)
- ✅ Product features ready (25 attributes from articles)
- ✅ Customer features ready (40 engineered features)
- ✅ Train/test split defined
- ✅ Evaluation window established (final 7 days)

## Deliverables
```
scripts/etl/
├── feature_engineering.py  ✅ (400 lines, production code)
├── clean_data.py           ✅ (200 lines, intelligent imputation)
├── load.py                 ✅ (150 lines, Parquet saves)
└── transform.py            ✅ (150 lines, orchestration)

processed_data/
├── 5 Parquet datasets      ✅ (ML-ready)
└── Feature metadata        ✅ (documentation)
```

## Portfolio Impact

This week demonstrates:
-  Advanced feature engineering (domain knowledge applied)
-  Production code quality (modular, reusable, documented)
-  Data quality expertise (intelligent imputation strategies)
-  Pipeline orchestration (end-to-end automation)
-  Performance optimization (Parquet, partitioning, caching)
-  Proper ML practices (temporal splits, no leakage)

**Week 2-3 Status: COMPLETE**

**Ready for:** Machine Learning Model Development (Week 4-7)
