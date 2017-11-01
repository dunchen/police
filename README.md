# Presentation Outline
## Overview of the Dataset 
- Mean of Crime Rate and other meaningful variables 
- Most frequent incident type 
- Etc… 
这里可以用画图辅助表示，更加直观一些

## Statement of the Problem 
Crime does not always happen at the same places. The crime rates strongly correlate with time of the day, date and location 
crime rates怎么求呀？这个我们用count／the number of crimes each day（quantitative）还是今天发生or没发生（categorical）variable？

- Option 1: Three pairs of maps：3组juxtaposition的图？
  - Same time of the day, same date, different location 
  - Same time of the day, different date, same location (Try 2017/06/26 and 2014/11/03)
  - Different time of the day, same date, same location
  
- Option 2: Linear Regression Model (Crime rate ~ time of the day + date)
这里的crime rate指的是什么？怎么算呢？
就是用R fit一个linear model？有点太简单了吧...
这个相关性可能没有多少，毕竟date的range太大了，time of the day的range也太大了

或者换成画图怎么样？ggplot之类的，把犯罪类型、数量依时间变化的趋势表示出来——这个用tableau可以做，也可以用R做。Tableau结果可能更fancy一点。————这个画图部分也可以移到最最最开头，一个data的overview。
你们有人可以安装tableau吗？sorry我的电脑内存不足orz。

As a consequence, the police might have trouble deciding where to patrol and the efficiency might be low.
要说明这一点的话可能还是option 1 更合适一点。
So our research problem: How to predict the patrol route and increase efficiency of the police
predict the patrol route我们应该还做不到，我们只能predict某事件在一些区域中发生的案件数（？）——这一预测结果的significance还需要解释一下（当展示结果时）

## Solution 
- We adopt LSTM model and give suggestion on the patrol routing.
  - Process (Choice of data, training set, test set) (used 281056 observation out of 1829230 observations)

- Using collaborating LSTM-controller network to predict future crime heat map from dynamically chosen past history


 - If we want to predict the crime heat map for a certain day, an obvious solution is to use extrapolation of certain number of days of heat map data in past history. But parameters of classical model for extrapolation is intensively hand-crafted and is heavily relied on intuition.  For instance, how many days of history should we take into account to predict the current day? If we consider a few days in the past, we would miss out the larger crime trend and long term causal relationship that might change the prediction dramatically from the local extrapolation. An example would be the organized crime or terrorist attack, the sign of which is long time planning/preparation and sudden explosion. The short term extrapolation would totally fail to give a warning to such explosion. But if we take into account too much past information in every case, we might be distracted in predicting short term change in crime pattern: for instance, a special event, like an concert, happened in the city, which would certainly cause dramatical short term change in crime pattern. Thus, the machine needs to automatically change its model of extrapolation based on its understanding of current situation. And such understanding of current situation and its relationship to suitable model of extrapolation must be also learned from past prediction experience. In short, the machine needs not only to learn how to do prediction but also learn to learn how to do prediction. In our project, we limit ourselves to teach machine to adjust number of days in the past to consider. To implement this, we have built two neural networks: the prediction network and the controller network. The prediction network is a LSTM (long-term-short-memory network), which learns to predict future crime heat map. Its number of days in the past to consider in each iteration is determined by the controller network. The controller network is a multi-layer perceptron, which takes as input the current state of prediction network (which represents the information gathered and estimation considering so far) and the and the next day the prediction network will take into account for the prediction. The controller network will decide whether should stop the prediction network to move further in the past history. Because we don’t give any a priori information to the machine, both of the networks are learning together from the beginning. We could see better controller network will certainly help prediction network to lower the loss in prediction by eliminating distracting history information. On the other hand, a better prediction network will provide more accurate and insightful information of the trend in history in its state representation, which in return will help the controller network to find the correct length of looking back that will better incorporate such trend for the prediction network to learn. Thus, they, learning together, will converge to an optimal solution.
  
## Result 
- Error decreases 
- Application : let’s look at an example- > Comparison of the prediction by mean and by LSTM 
    - Pick a specific shift, output the probability vector of that shift and location 
    - Graph 1: Crime hotspot prediction by mean 
    - Graph 2: Crime hotspot prediction by LSTM
    
## Discussion 
- Further work: Specify the type of incident in the training model 

+ Limitations & Future Exploration?
