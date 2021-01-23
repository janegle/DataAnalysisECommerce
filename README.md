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
<p style="font-family:Papyrus; font-size:4em;"> Among mobile users, there's statistically significant positive lift for Session Revenue (1.61%), CVR (1.81%) and ATC (0.61%) </p> 

![mobile](./images/mobile_users_ttest.png "Image1")

Analyzing results cumulatively:
- Given the short duration of the test, CVR performs the best with biggest lift at the end of the test period, most stable lift over time, and increasing t-stat over time
- Session revenue’s lift over time and t-stat over time both have U shape, indicating that customers need more time to adjust spending behavior
- ATC’s lift over time and t-stat over time are not stable

![mobile](./images/SessionRevenue_lift.png "Image1") ![mobile](./images/SessionRevenue_t-stat.png "Image1")

<p float = "left">
	<img src="./images/SessionRevenue_lift.png" /> 
	<img src="./images/SessionRevenue_lift.png" />
</p>

### Part 2: Fit a number of machine learning models to understand what factors in the dataset can best predict conversion rate

