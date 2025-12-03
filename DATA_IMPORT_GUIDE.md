# GitHub Data Import Guide

## Overview

Your 3i powered dashboard now includes a complete data import system that pulls CSV data from the GitHub cashew repository and stores it in your Supabase database for analysis and visualization.

## What Gets Imported

The system imports three types of data:

1. **Customers** (`customers.csv`)
   - Customer demographics (name, age, gender, location)
   - Loyalty program information (tier, points, value)
   - Contact information (email, phone)
   - Marketing preferences

2. **E-commerce Purchases** (`ecommerce_purchases.csv`)
   - Order transactions with timestamps
   - Sales channels (Shopee, Lazada, Website)
   - Payment methods
   - Revenue breakdown (subtotal, shipping, GST, total)

3. **Traffic Acquisition** (`traffic_acquisition.csv`)
   - Marketing campaign performance
   - Platform metrics (impressions, clicks, CTR)
   - Conversion data
   - Ad spend and ROI

## How to Use

### Step 1: Access the Data Import Tab

1. Open your dashboard
2. Click on the "Data & Analytics" tab in the navigation

### Step 2: Import Data

1. Select the data type you want to import:
   - **Import All Data** - Imports all three datasets at once (recommended for first import)
   - **Customers Only** - Imports just customer data
   - **E-commerce Purchases Only** - Imports just transaction data
   - **Traffic Acquisition Only** - Imports just marketing data

2. Click "Start Import" button

3. Wait for the import to complete (this may take a minute or two for large datasets)

4. Check the results summary to see how many records were imported

### Step 3: View Analytics

After importing data, scroll down to see:

- **Summary Metrics** - Total revenue, customers, and average order value
- **Channel Performance** - Revenue and metrics by sales channel
- **Customer Loyalty Tiers** - Performance breakdown by loyalty level
- **Marketing Campaign Performance** - Campaign effectiveness metrics
- **Monthly Revenue Trends** - Historical revenue patterns

### Step 4: Monitor Import History

The "Import History" section shows:
- All past imports with timestamps
- Import status (success, failed, partial)
- Number of records imported
- Real-time updates when new imports are running

## Database Structure

All imported data is stored in Supabase tables:

- `imported_customers` - Customer records
- `imported_ecommerce_purchases` - Order transactions
- `imported_traffic_acquisition` - Marketing campaign data
- `data_import_logs` - Import operation history

## Analytics Views

Pre-built database views provide instant analytics:

- `v_channel_performance` - Sales by channel
- `v_customer_loyalty_analysis` - Loyalty tier metrics
- `v_marketing_analytics` - Campaign performance
- `v_monthly_revenue_trends` - Time-based revenue
- `v_top_customers` - High-value customer identification
- `v_platform_roi` - Marketing ROI by platform

## Key Features

✅ **Automatic Updates** - Import updates existing records (upsert logic)
✅ **Data Validation** - Built-in error handling and validation
✅ **Batch Processing** - Handles large datasets efficiently
✅ **Real-time Updates** - Dashboard refreshes automatically
✅ **Audit Trail** - Complete history of all imports
✅ **No Manual Downloads** - Fetches directly from GitHub

## Important Notes

- The GitHub repository is only needed during the import process
- Once imported, all data lives in your Supabase database
- You can delete the GitHub repository after importing if desired
- Re-importing the same data will update existing records (no duplicates)
- The dashboard works entirely from Supabase after import

## Troubleshooting

**Import Failed**
- Check your internet connection
- Verify the GitHub repository is accessible
- Check the Import History for error details

**No Data Showing in Analytics**
- Ensure you've imported data first
- Refresh the page
- Check that records were successfully imported in Import History

**Slow Performance**
- Large datasets may take time to import
- Analytics views are optimized for fast queries
- Consider importing data types separately for better control

## Next Steps

After importing data, you can:
- Explore different analytics visualizations
- Use the AI Assistant to query the imported data
- Set up automated insights based on the imported metrics
- Create custom reports combining imported and existing dashboard data

---

For technical details about the implementation, see the database migrations in `supabase/migrations/`.
