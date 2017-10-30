# Presentation Outline
## Overview of the Dataset 
- Mean of Crime Rate and other meaningful variables 
- Most frequent incident type 
- Etc… 
## Statement of the Problem 
Crime does not always happen at the same places.The crime rate strongly correlates with time of the day,  date and location 
- Option 1: Three pairs of maps 
  - Same time of the day,  same date, different location 
  - Same time of the day, different date, same location (Try 2017/06/26 and 2014/11/03)
  - Different time of the day, same date, same location
- Option 2: Linear Regression Model (Crime rate ~ time of the day + date)

As a consequence, the police might have trouble deciding where to patrol and the efficiency might be low. 
SO Our research problem: How to predict the patrol route and increase efficiency of the police
## Solution 
- We adopt LSTM model and give suggestion on the patrol routing.
  - Process (Choice of data, training set, test set) (used 281056 observation out of 1829230 observations)
## Result 
- Error decreases 
- Application : let’s look at an example- > Comparison of the prediction by mean and by LSTM 
    - Pick a specific shift, output the probability vector of that shift and location 
    - Graph 1: Crime hotspot prediction by mean 
    - Graph 2: Crime hotspot prediction by LSTM
## Discussion 
- Further work: Specify the type of incident in the training model 
