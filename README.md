# Predicting Gentrification in Denver, CO

## [Read the full report here](https://github.com/thebbennett/predicting-gentrification/blob/master/Predicting%20Gentrification%20in%20Denver%2C%20CO.pdf)

## Methodology 
The process for this project is broken up into several parts:

1. Define gentrification
2. Determine scale of gentrification from 2011 to 2016
3. Hypothesize predictor variables for the year 2011 that might predict the outcome from the
above calculation
4. Gather the data for the predictor variables and wrangle it into Python
5. Determine correlation between predictor variables and outcome variable
6. Create a linear regression model from the data
7. Determine accuracy of linear regression model

## Definition of Gentrification 
In an early attempt at this project, I used the following definition from Wikipedia to determine
whether a census tract had gentrified from 2011 to 2016:

"Whether gentrification has occurred in a census tract in an urban area in the United States
during a particular 10-year period* between censuses can be determined by a method used in
a study by Governing:[50] If the census tract in a central city had 500 or more residents and at
the time of the baseline census:
* had median household income and median home value in the bottom 40th percentile
and at the time of the next 10-year census the
* tract's educational attainment (percentage of residents over age 25 with a bachelor's
degree) was in the top 33rd percentile;
* the median home value, adjusted for inflation, had increased;
* and the percentage of increase in home values in the tract was in the top 33rd
percentile when compared to the increase in other census tracts in the urban area
then it was considered to have been gentrified.

* I didn’t want you to overlook this crucial detail! Standard gentrification models look over a 10-year
period. I specifically chose to look at a 5-year window encompassing the time immediately before and
immediately after the legalization of weed in Colorado.

While this formula carries some significant gravitas behind it, I ultimately found that it was not a good
fit for analyzing Denver’s changing urban landscape. From my observations of Denver while living
here, this formula did not accurately classify census tracts as having gentrified or not. The above
formula had to be tweaked to be more lenient to classify any census tract as having gentrified
between 2011 and 2016.

Therefore, I decided to parse out one quantifiable component of gentrification. For the purposes of
this project, I chose to define gentrification in terms of the increase in rent from 2011 to 2016.
