# H&M E-commerce Dataset - Data Dictionary

## Dataset Overview
- **Source:** Kaggle H&M Personalized Fashion Recommendations
- **Date Range:** September 2018 - September 2020 (~2 years)
- **Total Size:** 3.5GB uncompressed
- **Total Records:** 33.2M rows across 3 tables

## Tables

### 1. articles.csv (Product Catalog)
- **Rows:** 105,542 products
- **Size:** 35MB
- **Columns:** 25
- **Primary Key:** article_id
- **Description:** Complete product catalog with detailed attributes

**Key Columns:**
- article_id: Unique product identifier
- product_type_name: Product category (Vest top, Dress, etc.)
- colour_group_name: Color category
- department_name: Department classification
- garment_group_name: Garment grouping

### 2. customers.csv (Customer Profiles)
- **Rows:** 1,371,980 customers
- **Size:** 198MB
- **Columns:** 7
- **Primary Key:** customer_id (hashed)
- **Description:** Customer demographic and preference data

**Key Columns:**
- customer_id: Unique customer identifier (hashed for privacy)
- club_member_status: Membership status (ACTIVE, etc.)
- age: Customer age
- postal_code: Location (hashed)

### 3. transactions_train.csv (Purchase History)
- **Rows:** 31,788,324 transactions
- **Size:** 3.3GB
- **Columns:** 5
- **Date Range:** 2018-09-20 to 2020-09-22
- **Description:** Complete purchase transaction history

**Key Columns:**
- t_dat: Transaction date
- customer_id: FK to customers
- article_id: FK to articles
- price: Purchase price (normalized 0-1)
- sales_channel_id: 1=Store, 2=Online

## Business Use Cases
1. **Churn Prediction:** Identify customers who haven't purchased in 90+ days
2. **Recommendation System:** Collaborative + content-based filtering
3. **Customer Segmentation:** RFM analysis + K-means clustering
4. **Revenue Optimization:** Identify high-value customers and products

## Data Quality Notes
- All customer_id and postal_code values are hashed for privacy
- Price values are normalized (0-1 range)
- No missing values in transaction fact table
- Date range: 24 months of complete history
