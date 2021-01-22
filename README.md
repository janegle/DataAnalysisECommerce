# DataAnalysisECommerce
This project aims to explore an e-commerce dataset containing information about visitors' activities during a seven-day period when an A/B test was conducted. 
Each row of data represents a session, and includes the following information:
1. Outcomes of interest
- Clicks: binary
- ATC (add-to-cart): binary
- CVR (conversion): binary
- SessionRevenue: integer
2. Visitors' information:
- SessionStartDate: date
- platform: categorical (mobile and desktop)
- visitorType: categorial (new, activated, returning, acquired)
- testGroup: categorical (control and variation)
- CategoryID (product types): categorical (TV Stands, sofas, area rugs, etc.)

### Part 1: Analyze whether the variation should be rolled out
![mobile](/images/mobile_ttest.png "T-test results for mobile users")
### Part 2: Fit a number of machine learning models to understand what factors in the dataset can best predict conversion rate

