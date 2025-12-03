# Role-Based Dashboard System - Setup Guide

## Overview

The application now features a comprehensive role-based authentication and authorization system with personalized dashboards for different departments:

- **C-Suite Executives** - Full access to all data and strategic overview
- **Sales Executives** - Sales-focused metrics, market opportunities, and customer insights
- **Operations Executives** - Inventory management, demand forecasting, and supply chain metrics
- **Finance Executives** - Financial performance, ROI analysis, and investment opportunities

## Features Implemented

### 1. Authentication System
- Email/password authentication using Supabase Auth
- Secure login and signup flows
- Session management with automatic token refresh
- Protected routes with role-based access control

### 2. User Profiles and Roles
Each user has a profile with:
- Full name
- Role: `c_suite`, `sales_exec`, `operations_exec`, or `finance_exec`
- Department: `Executive`, `Sales`, `Operations`, or `Finance`
- Personalization settings (future use)
- Preferred metrics (future use)

### 3. Department-Specific Dashboards

#### Sales Dashboard
- Sales KPIs (revenue, conversion rates, customer metrics)
- Market opportunities ranked by viability
- Sales-specific narrative insights
- Performance metrics table

#### Operations Dashboard
- Operations KPIs (inventory turnover, DSI, capacity utilization)
- **Demand Forecasting** with interactive charts
- Inventory recommendations by urgency level
- Supply chain and operations insights
- Toggle between weekly, monthly, and quarterly forecasts

#### Finance Dashboard
- Financial KPIs (profit margins, ROI, cash flow)
- Investment opportunities with ROI analysis
- Financial impact issues tracking
- Cross-department financial insights
- Comprehensive performance metrics

#### C-Suite Dashboard (Executive)
- Aggregated view of all departments
- Strategic insights and action scorecards
- Market entry viability index
- Data import and analytics tools
- AI assistant for conversational queries

### 4. Demand Forecasting System (Operations Only)

New database tables:
- `demand_forecasts` - Stores demand predictions with confidence intervals
- `inventory_recommendations` - AI-generated inventory optimization suggestions
- `forecast_accuracy_metrics` - Tracks forecasting model performance

Features:
- Multi-horizon forecasting (weekly, monthly, quarterly, yearly)
- Confidence interval visualization
- Actual vs predicted comparison
- Forecast accuracy tracking
- Product and market-specific predictions

### 5. Row Level Security (RLS)

All tables have RLS policies enforcing:
- Operations and C-Suite can access demand forecasting tables
- Sales users only see Sales department data
- Finance users have broader read access for analysis
- Users cannot access data from other departments
- C-Suite has full cross-department visibility

## Setting Up Users

### Step 1: User Signup
1. Users can sign up through the login page
2. Email and password are required
3. Full name is captured during signup

### Step 2: Configure User Profile
After a user signs up, an admin must configure their profile in the `user_profiles` table:

```sql
INSERT INTO user_profiles (id, full_name, role, department, preferred_metrics)
VALUES (
  'user-id-from-auth-users',
  'John Doe',
  'sales_exec',  -- or 'operations_exec', 'finance_exec', 'c_suite'
  'Sales',       -- or 'Operations', 'Finance', 'Executive'
  ARRAY['revenue', 'conversion_rate']
);
```

### Step 3: User Login
Once the profile is configured, users can:
1. Sign in with their credentials
2. Automatically be routed to their department dashboard
3. See only data relevant to their role
4. Access features appropriate to their department

## Dashboard Access Control

### Access Matrix

| Feature | C-Suite | Sales | Operations | Finance |
|---------|---------|-------|------------|---------|
| Executive Overview | ✅ | ❌ | ❌ | ❌ |
| Sales Dashboard | ✅ | ✅ | ❌ | ❌ |
| Operations Dashboard | ✅ | ❌ | ✅ | ❌ |
| Finance Dashboard | ✅ | ❌ | ❌ | ✅ |
| Demand Forecasting | ✅ | ❌ | ✅ | ❌ |
| All Department KPIs | ✅ | Own Only | Own Only | All (Read) |
| Market Entry Data | ✅ | ✅ | ✅ | ✅ |
| RCA Analyses | ✅ | Own Dept | Own Dept | All (Read) |

## Testing the System

### Create Test Users

Create test users for each role:

```sql
-- Sales Executive
INSERT INTO user_profiles (id, full_name, role, department)
VALUES ('uuid-1', 'Sarah Sales', 'sales_exec', 'Sales');

-- Operations Executive
INSERT INTO user_profiles (id, full_name, role, department)
VALUES ('uuid-2', 'Oliver Operations', 'operations_exec', 'Operations');

-- Finance Executive
INSERT INTO user_profiles (id, full_name, role, department)
VALUES ('uuid-3', 'Frank Finance', 'finance_exec', 'Finance');

-- C-Suite Executive
INSERT INTO user_profiles (id, full_name, role, department)
VALUES ('uuid-4', 'Charlie CEO', 'c_suite', 'Executive');
```

### Verify Access Control

1. Sign in as each user type
2. Verify you only see your department's dashboard
3. Try to access data from other departments (should be blocked by RLS)
4. Test demand forecasting (only Operations and C-Suite should access)

## Populating Demand Forecast Data

To populate the demand forecasting system with sample data:

```sql
-- Sample demand forecasts
INSERT INTO demand_forecasts (
  product_code,
  country_code,
  forecast_period_start,
  forecast_period_end,
  forecast_horizon,
  predicted_demand,
  confidence_lower,
  confidence_upper,
  confidence_level
)
VALUES
  ('PROD-001', 'SG', '2025-01-01', '2025-01-31', 'monthly', 5000, 4500, 5500, 95),
  ('PROD-001', 'SG', '2025-02-01', '2025-02-28', 'monthly', 5200, 4600, 5800, 95),
  ('PROD-002', 'MY', '2025-01-01', '2025-01-31', 'monthly', 3000, 2700, 3300, 95);

-- Sample inventory recommendations
INSERT INTO inventory_recommendations (
  product_code,
  country_code,
  recommendation_type,
  current_quantity,
  recommended_quantity,
  forecast_demand,
  safety_stock,
  urgency_level,
  reasoning
)
VALUES
  (
    'PROD-001',
    'SG',
    'reorder',
    1200,
    5000,
    5000,
    500,
    'high',
    'Current inventory will run out in 7 days based on forecasted demand. Recommend immediate reorder.'
  );
```

## Security Best Practices

1. **Never expose user credentials** - Use Supabase Auth for secure authentication
2. **Always use RLS policies** - Database-level security prevents unauthorized access
3. **Validate user roles** - Check roles on both client and server side
4. **Audit access logs** - Monitor who accesses what data
5. **Least privilege principle** - Users only see what they need for their job

## Next Steps

To further enhance the system:

1. **User Management Interface** - Build admin UI to manage user roles and departments
2. **Dashboard Customization** - Allow users to customize widget layouts
3. **Advanced Forecasting** - Implement ML-based demand prediction algorithms
4. **Notifications** - Add role-based alerts and notifications
5. **Audit Trail** - Track all data access and modifications
6. **Export Controls** - Add role-based restrictions on data exports
7. **Mobile Responsiveness** - Optimize dashboards for mobile devices

## Troubleshooting

### User sees "Authentication Required"
- User is not signed in
- Session has expired
- Check authentication status in browser console

### User sees "Access Denied"
- User profile not configured in `user_profiles` table
- User role doesn't match required role for the route
- RLS policy blocking access

### Demand forecasts not showing
- User is not Operations or C-Suite role
- No data in `demand_forecasts` table
- RLS policy blocking access

### Build errors
- Run `npm install` to ensure all dependencies are installed
- Check TypeScript errors with `npm run typecheck`
- Ensure all imports are correct

## Support

For issues or questions:
1. Check the browser console for errors
2. Verify RLS policies in Supabase dashboard
3. Test authentication flow with different user roles
4. Review the network tab for failed API requests
