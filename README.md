# Presentation Outline
## Overview of the Dataset  
- Most frequent incident type (Not all are valuable) 
- Etc… 
这里可以用画图辅助表示，更加直观一些

## Statement of the Problem 
Quesiton for the police: when and where to patrol. 
Hypothesis: Crime does not always happen at the same places. The probability of having a crime incident strongly correlates with time of the day, date and location

- Option 1: Three pairs of maps：
  - Same time of the day, same date, different location 
  - Same time of the day, different date, same location (Try 2017/06/26 and 2014/11/03)
  - Different time of the day, same date, same location
  
- Option 2: Linear Regression Model (Crime probability ~ time of the day + date)

As a consequence, the police might have trouble deciding where to patrol and the efficiency might be low.
So our research problem: How to predict the patrol route and increase efficiency of the police
predict the patrol route

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
    - Graph 2: Crime hotspot prediction by LSTM (significance in respect to time)
    
## Discussion 
- Further work: Specify the type of incident in the training model 

+ Limitations & Future Exploration?

- Possion 
- Wheather there is spatial correlation







# 11.01 17:23 edit
我们的故事中需要涉及／圆回来的问题：

### 1. 背景介绍
1.1 Cincinnati警察局的dataset的variable们，数据是怎样收集的，cases都是什么etc， 要求／希望我们给出什么建议

1.2 一些我们接下来会用到的variable的解释：xx代表yy

eg. fact statement: patrol time是什么，我们怎么用的

### 2. 展示data
2.1 bar plot/ histogram: x-axis是所有的（？）incident type, y-axis是dataset种的个数／count，图里面可以按照数目多少把案件类型从大到小排序。
－>说明有些type发生得更frequently。

2.2 Three pairs of maps：
  (a) Same time of the day, same date, different location 
  (b) Same time of the day, different date, same location (Try 2017/06/26 and 2014/11/03) (用不同的颜色标记data so that the info can be visualized on a single graph)
  (c) Different time of the day, same date, same location (same as above: can use 3 colors for 3 shifts)

2.3 从上面3个图我们可以看到，
time of the day matters, (///但是这个的预测我们没有办法做到精确，只能用shifts代替；所以这里的time of the day我们实际上指的是different shifts?)
date matters, 
region matters, in terms of the number of incidents

### 3. 我们对这个情况的分析
3.1 我们的（假设）／分析：一个police officer wants to get to the place where crime incidents occur as fast as possible / wants to cover as many of cases in which his or her presence could help, so that the problems could be solved more efficiently.
Thus, the police might have trouble deciding where to patrol during his or her shift
-> 2 variables: shift & region

3.2 -> research problem: How to predict the patrol route and increase efficiency of the police
（????predict the patrol route这一点怎么说存疑）

3.3 关于我们的data：
因为我们的研究问题是 为警察巡逻的时间、地点提出建议，所以not all incident types matter.
-> we select some of the incident types for which a shorter time that police officers take to be present on the sites is important / has an influence on the resolutions of the problems. 
Thus, we focus on 4 types: car_accident .etc
+ 最好可以draw connection, 说一下这几个案件也是发生非常频繁, and constitute a significant part of a police's job.

### 4. 数据分析
4.1 Poisson model 

4.1.1 找出合适的variables which become our response and explanatory variables
（这些variable可以与LSTM相同也可以不同）

4.1.2 建立model：model本身的function

4.1.3 展示结果

4.1.4 展示结果的有限性：
(a) assume independence among regions / independence between variables (?????存疑，需要找一下书或者其他解释为什么它是assume independence的)
(b) result/prediction: fixed (?????)
(c) look at the R^2 or some other measurement to show how well the model captures the variations in data.

4.2 LSTM

4.2.1 LSTM的介绍：是什么

4.2.2 LSTM在我们的case中的应用：variables的选取，怎么预测

4.2.3 present结果：
画图表示：选择不同的date预测结果不一样

4.2.4 LSTM的优越性，与Poisson model相比
（+说明一下Poisson最普遍应用的、合适这个情况的统计模型了，所以4.2里面和它比较是有意义的）
eg. 是动态的，自动学习，所以可以规避xxxyyyzzz Poisson capture不到的情况
eg. 不同的date预测结果不一样
eg. 不assume independence

### 5. 所以我们的结果非常好非常厉害。
