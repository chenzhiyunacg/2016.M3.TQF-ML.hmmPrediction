# 2016.M3.TQF-ML.hmmPrediction
**Project description**

Hidden Markov model was first applied to speech recognition, and then in other areas have been widely used, such as natural language processing, biological gene sequence analysis and face recognition and so on. This project tries to apply hidden markov model in stock market's prediction. See the [proposal](https://github.com/chenzhiyunacg/2016.M3.TQF-ML.hmmPrediction/blob/master/Proposal_Chen%20Zhiyun.pdf).

**Features**

lag of trading by daily data

trading fee

**Methods**

1.identify hidden state

2.select indicators

3.calculate evaluation indicators

**Data**

see [data](https://github.com/chenzhiyunacg/2016.M3.TQF-ML.hmmPrediction/blob/master/indicators.csv)

I extract data from Wind database.

Including Open, close, high, low of Shanghai Stock composite index and other index such as VMA, RSI, PVT.. 

Daily data from Jan 4, 2000 to Dec 31, 2016 

**Implementation**

[hmm_ML.ipynb](https://github.com/chenzhiyunacg/2016.M3.TQF-ML.hmmPrediction/blob/master/hmm_ML.ipynb)

[hmm_ML.py](https://github.com/chenzhiyunacg/2016.M3.TQF-ML.hmmPrediction/blob/master/hmm_ML.py)

**Conclusion**

See [presentation](https://github.com/chenzhiyunacg/2016.M3.TQF-ML.hmmPrediction/blob/master/Final%20presentation.pdf).

The HMM model based on the daily return has such a high yield and high winning rate because of its ability to form recognition and automatic switching strategy. 

In trend market, like many momentum strategies, the model can seize the trend in the case of reducing the frequency of transactions, as much as possible to share the benefits of the entire trend.

In the shock market, our model can acquire short-term or even shortshort-term gains through increasing the number of transactions and improving profit to loss ratio and winning probability.
