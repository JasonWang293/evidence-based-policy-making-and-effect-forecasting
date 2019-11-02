import pandas as pd
from pandas import DataFrame

#Policy effect analysis, counting from 0
def policy_effect_analysis(score_df,time_granularity,start_position=0):
    effect_list=[]
    time_index_list=[]
    if time_granularity=='month':
        for time_index in range(start_position,len(score_df.index)):
            effect_list.append(((score_df.iloc[time_index]-score_df.iloc[time_index-12])/
                                score_df.iloc[time_index-12]).round(2))
        time_index_list=score_df.index[start_position:len(score_df.index)]
    if time_granularity=='year':
        for time_index in range(start_position,len(score_df.index)):
            effect_list.append(((score_df.iloc[time_index]-score_df.iloc[time_index-1])
                                /score_df.iloc[time_index-1]).round(2))
        time_index_list=score_df.index[start_position:len(score_df.index)]
    effect_list=DataFrame(effect_list,index=time_index_list,
                          columns=['positive','negative','final'])
    return effect_list

contents_sentiment_score_df=pd.read_csv(open('sentiment analysis results with month time format.csv',encoding='utf-8'))
effect_df=policy_effect_analysis(
        contents_sentiment_score_df.groupby(contents_sentiment_score_df['time']).sum(),
        'month',18)
effect_df.to_csv('policy effect analysis by month.csv')
contents_sentiment_score_df.groupby(contents_sentiment_score_df['time']).sum().\
    to_csv('sentiment score summed by month.csv')
contents_sentiment_score_df=pd.read_csv(open('sentiment analysis results with year time format.csv',encoding='utf-8'))
effect_df=policy_effect_analysis(
        contents_sentiment_score_df.groupby(contents_sentiment_score_df['time']).sum(),
        'year',1)
effect_df.to_csv('policy effect analysis by year.csv')
contents_sentiment_score_df.groupby(contents_sentiment_score_df['time']).sum().\
    to_csv('sentiment score summed by year.csv')
