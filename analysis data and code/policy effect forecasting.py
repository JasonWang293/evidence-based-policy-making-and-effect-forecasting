import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
import random

#Calculate the cosine similarity of cluster centroids, and the policy_centers_index is counted from 1.
def cos_similarity_cal(clusters_center,policy_centers_index):
    clusters_cos_sim_list=[]
    for cluster_center in clusters_center:
        cos_sim_list=[]
        for policy_center_index in policy_centers_index:
            cos_sim=np.dot(cluster_center,clusters_center[policy_center_index-1])/(
                np.sqrt(sum(np.square(cluster_center)))*
                np.sqrt(sum(np.square(clusters_center[policy_center_index-1]))))
            cos_sim_list.append(cos_sim)
        clusters_cos_sim_list.append(max(cos_sim_list))
    return clusters_cos_sim_list

#Calculate the policy relevance of each cluster
def policy_relevance_cal(cos_sim_list,policy_centers_index,amplify_coefficient=1.2):
    policy_rel_list=[]
    for cos_sim_index in range(0,len(cos_sim_list)):
        if (cos_sim_index+1) in policy_centers_index:
            policy_rel_list.append(cos_sim_list[cos_sim_index])
        else:
            policy_rel_list.append(cos_sim_list[cos_sim_index]*amplify_coefficient)
    return policy_rel_list

#Calculate the attitude of public
def attitude_cal(sen_score_df):
    att_list=[]
    pos_score_list=list(sen_score_df['positive'].sort_values(ascending=True))
    neg_score_list=list((sen_score_df['negative']*(-1)).sort_values(ascending=True))
    cons_num=len(sen_score_df.index)
    for sen_score_index in sen_score_df.index:
        neg_per=(neg_score_list.index(sen_score_df.ix[sen_score_index]['negative']*(-1))+1)/cons_num
        if neg_per>0.7:
            att_list.append(1.7-neg_per)
        else:
            pos_per=(pos_score_list.index(sen_score_df.ix[sen_score_index]['positive'])+1)/cons_num
            if pos_per>0.7:
                att_list.append(pos_per+0.3)
            else:
                att_list.append(1.0)
    return att_list

#Upload data form txt file as a whole
def read_txt_file_whole(filename,by_list=True):
    data_file=open(filename+'.txt','r',encoding='utf-8')
    if by_list:
        data_list=data_file.read().lstrip('[').rstrip(']').split(', ')
    else:
        data_list=data_file.read()
    return data_list

cluster_num=18#The number of clusters
Ipe=9#Intuitive of policy effects
Pe=8.5#Propaganda effect
Spi=9#Success of policy implementation
Gc=0.85#Government credibility
Pra=1.2#Policy relevance amplification factor
policy_index_list=[5,9,10,12,13]#The corresponding cluster index of proposed policy, that is counted from 1.
clusters_center_df=pd.read_csv(open('tf-idf of each centroid.csv',encoding='utf-8'))
cos_similarity_list=cos_similarity_cal(clusters_center_df.values,policy_index_list)
policy_relevance_list=policy_relevance_cal(cos_similarity_list,policy_index_list,amplify_coefficient=Pra)
sentiment_score_df=pd.read_csv(open('sentiment analysis results with month time format.csv',encoding='utf-8'))
contents_cluster_index=read_txt_file_whole('km_clusters_list')
#Calculate the attitude of public
attitude_list=attitude_cal(sentiment_score_df)
sentiment_score_df['attitude']=attitude_list
implemented_sentiment_score_list=[]#Save the predicted sentiment score after the implementation of proposed policy
for cluster_index in range(cluster_num):
    Pr=policy_relevance_list[cluster_index]#Policy relevance
    Pkr=(max(Ipe,Pe)+10*Pr)/20#Policy known rate
    cluster_sentiment_score_df=sentiment_score_df[Series(contents_cluster_index)==str(cluster_index)]
    for content_index in cluster_sentiment_score_df.index:
        implemented_sentiment_score=[]
        #According to the Pkr of each cluster, determine whether the sentiment score of each discussion content would change with the roulette method.
        if random.uniform(0,1)<=Pkr:
            #Calculate the sentiment change degree of the content
            Scd=(Pra*Pr+(Spi+max(Ipe,Gc*Pe)/20))*\
                cluster_sentiment_score_df.ix[content_index]['attitude']
            #According to the Scd, calculate the positive and negative sentiment score after the implementation of proposed policy
            implemented_sentiment_score.append(
                    cluster_sentiment_score_df.ix[content_index]['positive']*Scd)
            implemented_sentiment_score.append(
                    cluster_sentiment_score_df.ix[content_index]['negative']/Scd)
            implemented_sentiment_score.append(implemented_sentiment_score[0]+implemented_sentiment_score[1])
            implemented_sentiment_score.append(cluster_sentiment_score_df.ix[content_index]['time'])
            implemented_sentiment_score.append(content_index)
        else:
            implemented_sentiment_score.append(cluster_sentiment_score_df.ix[content_index]['positive'])
            implemented_sentiment_score.append(cluster_sentiment_score_df.ix[content_index]['negative'])
            implemented_sentiment_score.append(cluster_sentiment_score_df.ix[content_index]['final'])
            implemented_sentiment_score.append(cluster_sentiment_score_df.ix[content_index]['time'])
            implemented_sentiment_score.append(content_index)
        implemented_sentiment_score_list.append(implemented_sentiment_score)

implemented_sentiment_score_df=DataFrame(implemented_sentiment_score_list,
                                         columns=['positive','negative','final','time','content_index'])
implemented_sentiment_score_df=implemented_sentiment_score_df.set_index('content_index')
implemented_sentiment_score_df=implemented_sentiment_score_df.sort_index(ascending=True)

#Sum the predicted sentiment score by month
implemented_month_sentiment_score_df=implemented_sentiment_score_df.groupby('time').sum()
month_sentiment_score_df=sentiment_score_df.groupby('time').sum()
policy_effect_month_df=DataFrame([implemented_month_sentiment_score_df['positive'],
                                 month_sentiment_score_df['positive'],
                                 implemented_month_sentiment_score_df['negative'],
                                 month_sentiment_score_df['negative'],
                                 implemented_month_sentiment_score_df['final'],
                                 month_sentiment_score_df['final']]).T
policy_effect_month_df.columns=['implemented_positive','original_positive',
                                'implemented_negative','original_negative',
                                'implemented_final','original_final']
policy_effect_month_df.to_csv('policy effect prediction by month'+str(policy_index_list)+'.csv')

#Compare the sum, average and median of sentiment score before and after the implementation of proposed policy
policy_effect={}
policy_effect_positive=[]
policy_effect_negative=[]
policy_effect_final=[]
policy_effect_index=['implemented_sum','original_sum','sum_change','sum_change_rate',
                     'implemented_mean','original_mean',
                     'mean_change','mean_change_rate',
                     'implemented_median','original_median',
                     'median_change','median_change_rate']
policy_effect_positive.append(implemented_sentiment_score_df['positive'].sum())
policy_effect_positive.append(sentiment_score_df['positive'].sum())
policy_effect_positive.append(policy_effect_positive[0]-policy_effect_positive[1])
policy_effect_positive.append(policy_effect_positive[2]/policy_effect_positive[1])
policy_effect_positive.append(implemented_sentiment_score_df['positive'].mean())
policy_effect_positive.append(sentiment_score_df['positive'].mean())
policy_effect_positive.append(policy_effect_positive[4]-policy_effect_positive[5])
policy_effect_positive.append(policy_effect_positive[6]/policy_effect_positive[5])
policy_effect_positive.append(implemented_sentiment_score_df['positive'].median())
policy_effect_positive.append(sentiment_score_df['positive'].median())
policy_effect_positive.append(policy_effect_positive[8]-policy_effect_positive[9])
policy_effect_positive.append(policy_effect_positive[10]/policy_effect_positive[9])

policy_effect_negative.append(implemented_sentiment_score_df['negative'].sum())
policy_effect_negative.append(sentiment_score_df['negative'].sum())
policy_effect_negative.append(policy_effect_negative[0]*(-1)-policy_effect_negative[1]*(-1))
policy_effect_negative.append(policy_effect_negative[2]/policy_effect_negative[1]*(-1))
policy_effect_negative.append(implemented_sentiment_score_df['negative'].mean())
policy_effect_negative.append(sentiment_score_df['negative'].mean())
policy_effect_negative.append(policy_effect_negative[4]*(-1)-policy_effect_negative[5]*(-1))
policy_effect_negative.append(policy_effect_negative[6]/policy_effect_negative[5]*(-1))
policy_effect_negative.append(implemented_sentiment_score_df['negative'].median())
policy_effect_negative.append(sentiment_score_df['negative'].median())
policy_effect_negative.append(policy_effect_negative[8]*(-1)-policy_effect_negative[9]*(-1))
policy_effect_negative.append(policy_effect_negative[10]/policy_effect_negative[9]*(-1))

policy_effect_final.append(implemented_sentiment_score_df['final'].sum())
policy_effect_final.append(sentiment_score_df['final'].sum())
policy_effect_final.append(policy_effect_final[0]-policy_effect_final[1])
policy_effect_final.append(policy_effect_final[2]/policy_effect_final[1])
policy_effect_final.append(implemented_sentiment_score_df['final'].mean())
policy_effect_final.append(sentiment_score_df['final'].mean())
policy_effect_final.append(policy_effect_final[4]-policy_effect_final[5])
policy_effect_final.append(policy_effect_final[6]/policy_effect_final[5])
policy_effect_final.append(implemented_sentiment_score_df['final'].median())
policy_effect_final.append(sentiment_score_df['final'].median())
policy_effect_final.append(policy_effect_final[8]-policy_effect_final[9])
policy_effect_final.append(policy_effect_final[10]/policy_effect_final[9])
policy_effect_df=DataFrame([policy_effect_positive,policy_effect_negative,
                            policy_effect_final],
                           index=['positive','negative','final'],
                           columns=policy_effect_index).T
policy_effect_df=policy_effect_df.round(2)
policy_effect_df.to_csv('descriptive statistics of policy effect forecasting results'+str(policy_index_list)+'.csv')