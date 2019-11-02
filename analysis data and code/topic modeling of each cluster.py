import jieba
import jieba.analyse
import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
from scipy.sparse import csr_matrix
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import Counter

#Upload content from file
def get_contents(file_name):
    contents_file=pd.read_csv(open(file_name+'.csv',encoding='utf-8'))
    contents_list=[]
    for i in range(len(contents_file.index)):
        contents_list.append(contents_file.ix[i]['ans_con'].strip())
    return contents_list,contents_file

#Delete html labels in the content
def del_html_label(contents_list):
    contents_nolabel_list=[]
    for content in contents_list:
        contents_nolabel_list.append(re.sub(r'<(.*?)>','',content))
    return contents_nolabel_list

#Text segmentation
def get_contents_token_list(contents_list):
    contents_token_list=[]
    for content in contents_list:
        contents_token_list.append(jieba.lcut(content))
    return contents_token_list

#Delete blank characters in the content
def del_blank_character(contents_token_list):
    contents_token_noblank_list=[]
    for content in contents_token_list:
        content_noblank=[]
        for token in content:
            token=token.strip()
            if token:
                content_noblank.append(token)
        contents_token_noblank_list.append(content_noblank)
    return contents_token_noblank_list

#Upload stopwords
def get_stopwords(file_name):
    stopwords=[str(line.encode('utf-8').decode('utf-8-sig').strip()) for line in open(file_name,'r',encoding='utf-8')]
    return stopwords

#Delete stopwords in the content
def del_stopwords(contents_token_list,stopwords):
    cleaned_list=[]
    for content in contents_token_list:
        operate_list=[]
        operate_list.extend(content)
        for word_token in content:
            if word_token in stopwords:
                operate_list.remove(word_token)
        cleaned_list.append(operate_list)
    return cleaned_list

#Connect the tokens with blank
def get_joined_content_tokens(contents_token_list):
    joined_content_tokens_list=[]
    for content in contents_token_list:
        joined_content_tokens_list.append(' '.join(content))
    return joined_content_tokens_list

#Generate the TF-IDF vector matrix
def tfidf_extractor(contents_list,min_df_sh=1,max_df_sh=1.0,ngram_r=(1,1),idf_use=True):
    vectorizer=TfidfVectorizer(analyzer=lambda x:x.split(' '),
                               min_df=min_df_sh,
                               norm='l2',
                               smooth_idf=True,
                               use_idf=idf_use,
                               ngram_range=ngram_r,
                               max_df=max_df_sh)
    features=vectorizer.fit_transform(contents_list)
    feature_names=vectorizer.get_feature_names()
    return vectorizer,features,feature_names

#Delete specific character in the TF-IDF vector matrix
def tfidf_del_element(tfidf_matrix,tfidf_names,del_element):
    if del_element in tfidf_names:
        element_index=tfidf_names.index(del_element)
        tfidf_dense=tfidf_matrix.todense()
        tfidf_dense=np.delete(tfidf_dense,element_index,axis=1)
        tfidf_matrix=csr_matrix(tfidf_dense)
        tfidf_names.remove(del_element)
    return tfidf_matrix,tfidf_names

#Display details of extracted content features
def display_features(features,feature_names,file_name=None):
    df=pd.DataFrame(data=features,columns=feature_names)
    if file_name:
        df.to_csv(file_name+'.csv')
    return df

#Topic modeling with NMF
def nmf_topic_model_get(tfidf_features,total_topics=2,max_iter_num=100):
    nmf=NMF(n_components=total_topics,max_iter=max_iter_num,alpha=0.1,l1_ratio=0.5)
    nmf.fit(tfidf_features)
    weights=nmf.components_
    return weights

#Get topic terms and its weight
def get_topics_terms_weights(weights,feature_names):
    feature_names=np.array(feature_names)
    sorted_indices=np.array([list(row[::-1])
                             for row in np.argsort(np.abs(weights))])
    sorted_weights= np.array([list(wt[index])
                             for wt,index in zip(weights,sorted_indices)])
    sorted_terms=np.array([list(feature_names[row])
                          for row in sorted_indices])
    topics=[np.vstack((terms.T,term_weights.T)).T
            for terms,term_weights in zip(sorted_terms,sorted_weights)]
    return topics

#Output the topic modeling results
def output_topics(topics,total_topics=1,weight_threshold=0.0001,
                  display_weights=False,num_terms=None,print_detail=False):
    topic_detail_list=[]
    for index in range(total_topics):
        topic=topics[index]
        topic=[(term,float(wt))
               for term,wt in topic]
        topic=[(word,round(wt,2))
               for word,wt in topic
               if abs(wt)>=weight_threshold]
        if num_terms:
            topic_detail_list.append(topic[0:num_terms])
        else:
            topic_detail_list.append(topic)
        if print_detail:
            if display_weights:
                print('Topic #'+str(index+1)+'with weights')
                print(topic[:num_terms] if num_terms else topic)
            else:
                print('Topic #'+str(index+1)+'without weights')
                tw=[term for term,wt in topic]
                print(tw[:num_terms] if num_terms else tw)
    return topic_detail_list

#Get the context keywords of topic terms
def topics_keywords_context_tokens_get(topics,contents_token_list, tfidf_feature_names,
                                       position=1,if_joined=False):
    topics_keywords_context_tokens_list=[]#Save the context keywords of all topics
    for topic in topics:
        topic_keywords_context_list=[]#Save the context keywords of one topic
        for keyword in topic:
            above_list=[]#Save the preceding context keywords of topic term
            below_list=[]#Save the following context keywords of topic term
            for content in contents_token_list:
                content=content.split(' ')
                keyword_index=np.argwhere(np.array(content)==keyword[0])#Get the index of topic term
                if len(keyword_index):#If the length of list is more than 0, then this content includes the topic term
                    for index in keyword_index:#Traverse the indexes of topic term, to get its context words
                        if (index[0]-position)>=0:#Get the preceding context words
                            if content[(index[0]-position)] in tfidf_feature_names:
                                above_list.append(content[(index[0]-position)])
                        if (index[0]+position)<len(content):#Get the following context words
                            if content[index[0]+position] in tfidf_feature_names:
                                below_list.append(content[index[0]+position])
            topic_keywords_context_list.append((keyword[0],keyword[1],Counter(above_list),Counter(below_list)))
        topics_keywords_context_tokens_list.append(topic_keywords_context_list)
    return topics_keywords_context_tokens_list

#Calculate the context weight
def topics_keywords_context_weights_get(context_list_size1,context_list_size2,context_list_size3,
                                        topics_tokens_names,topics_tokens_scores,context_num=5,
                                        position1_weight=1,position2_weight=0.9,
                                        position3_weight=0.8,
                                        context_word_count_weight=1,topic_weight=1):
    topics_scores_df=DataFrame(topics_tokens_scores,columns=topics_tokens_names)
    topics_keywords_context_tokens_list=[]#Save the context keywords of all topics
    for topic_index in range(len(context_list_size1)):
        topic_keywords_context_list=[]#Save the context keywords of one topic
        for keyword_index in range(len(context_list_size1[topic_index])):
            #Calculate the context weight of preceding keywords
            above_word_scores=Series()
            for above_word,above_word_count in context_list_size1[topic_index][keyword_index][2].items():#Traverse the preceding keywords of topic term
                if topics_scores_df.ix[topic_index][above_word]<0.0001:
                    #If the topic modeling weight of context words is less than 0.0001, set it to 0.00003 as a penalty
                    above_word_score=position1_weight*context_word_count_weight*above_word_count*\
                                     topic_weight*0.00003
                else:
                    above_word_score=position1_weight*context_word_count_weight*above_word_count*\
                                     topic_weight*topics_scores_df.ix[topic_index][above_word]
                above_word_scores[above_word]=above_word_score
            for above_word,above_word_count in context_list_size2[topic_index][keyword_index][2].items():
                if topics_scores_df.ix[topic_index][above_word]<0.0001:
                    above_word_score=position2_weight*context_word_count_weight*above_word_count*\
                                     topic_weight*0.00003
                else:
                    above_word_score=position2_weight*context_word_count_weight*above_word_count*\
                                     topic_weight*topics_scores_df.ix[topic_index][above_word]
                if above_word not in above_word_scores:
                    above_word_scores[above_word]=above_word_score
                else:
                    above_word_scores[above_word]=above_word_scores[above_word]+above_word_score
            for above_word,above_word_count in context_list_size3[topic_index][keyword_index][2].items():
                if topics_scores_df.ix[topic_index][above_word]<0.0001:
                    above_word_score=position3_weight*context_word_count_weight*above_word_count*\
                                     topic_weight*0.00003
                else:
                    above_word_score=position3_weight*context_word_count_weight*above_word_count*\
                                     topic_weight*topics_scores_df.ix[topic_index][above_word]
                if above_word not in above_word_scores:
                    above_word_scores[above_word]=above_word_score
                else:
                    above_word_scores[above_word]=above_word_scores[above_word]+above_word_score
            #Calculate the context weight of following keywords
            below_word_scores=Series()
            for below_word,below_word_count in context_list_size1[topic_index][keyword_index][3].items():#Traverse the following keywords of topic term
                if topics_scores_df.ix[topic_index][below_word]<0.0001:
                    below_word_score=position1_weight*context_word_count_weight*below_word_count*\
                                     topic_weight*0.00003
                else:
                    below_word_score=position1_weight*context_word_count_weight*below_word_count*\
                                     topic_weight*topics_scores_df.ix[topic_index][below_word]
                below_word_scores[below_word]=below_word_score
            for below_word,below_word_count in context_list_size2[topic_index][keyword_index][3].items():
                if topics_scores_df.ix[topic_index][below_word]<0.0001:
                    below_word_score=position2_weight*context_word_count_weight*below_word_count*\
                                     topic_weight*0.00003
                else:
                    below_word_score=position2_weight*context_word_count_weight*below_word_count*\
                                     topic_weight*topics_scores_df.ix[topic_index][below_word]
                if below_word not in below_word_scores:
                    below_word_scores[below_word]=below_word_score
                else:
                    below_word_scores[below_word]=below_word_scores[below_word]+below_word_score
            for below_word,below_word_count in context_list_size3[topic_index][keyword_index][3].items():
                if topics_scores_df.ix[topic_index][below_word]<0.0001:
                    below_word_score=position3_weight*context_word_count_weight*below_word_count*\
                                     topic_weight*0.00003
                else:
                    below_word_score=position3_weight*context_word_count_weight*below_word_count*\
                                     topic_weight*topics_scores_df.ix[topic_index][below_word]
                if below_word not in below_word_scores:
                    below_word_scores[below_word]=below_word_score
                else:
                    below_word_scores[below_word]=below_word_scores[below_word]+below_word_score
            topic_keywords_context_list.append((context_list_size1[topic_index][keyword_index][0],
                                               context_list_size1[topic_index][keyword_index][1],
                                                above_word_scores.sort_values(ascending=False)[0:min(len(above_word_scores),context_num)].round(3).to_dict(),
                                                below_word_scores.sort_values(ascending=False)[0:min(len(above_word_scores),context_num)].round(3).to_dict()))
        topics_keywords_context_tokens_list.append(topic_keywords_context_list)
    return topics_keywords_context_tokens_list

#Upload data from txt file as a whole
def read_txt_file_whole(filename,by_list=True):
    data_file=open(filename+'.txt','r',encoding='utf-8')
    if by_list:
        data_list=data_file.read().lstrip('[').rstrip(']').split(', ')
    else:
        data_list=data_file.read()
    return data_list

#Text segmentation and cleaning
jieba.load_userdict('segmentation.txt')
contents_list,contents_file=get_contents('Zhihu answer contents')
contents_nolabel_list=del_html_label(contents_list)
contents_token_list=get_contents_token_list(contents_nolabel_list)
contents_token_list=del_blank_character(contents_token_list)
joined_contents_token_list=get_joined_content_tokens(contents_token_list)
stopwords_list=get_stopwords('stopwords.txt')
contents_token_cleaned_list=del_stopwords(contents_token_list,stopwords_list)
joined_contents_token_cleaned_list=get_joined_content_tokens(contents_token_cleaned_list)
#Topic modeling
#Parameters
clusters_num=18
min_df_thre=2
max_df_thre=0.9
total_topics_num=2
ngram_r=(1,1)
context_num=5#The number of context keywords
topic_feature_display_num=15
cluster_topic_df=DataFrame()#Save the topic modeling results
cluster_topic_with_context=DataFrame()#Save the topic modeling results with context keywords
#Upload the cluster index of each content
file_name='km_clusters'
topic_model_name='nmf'
clusters=read_txt_file_whole(file_name+'_list')
#Perform the topic modeling in each cluster
for cluster_index in range(0,clusters_num):
    cluster_contents_token_list=list(Series(joined_contents_token_cleaned_list)[Series(clusters)==str(cluster_index)])
    #Relax the restriction of threshold if contents number of the cluster is to little.
    if len(cluster_contents_token_list)<20:
        min_df_threshold=1
        max_df_threshold=1.0
    else:
        min_df_threshold=min_df_thre
        max_df_threshold=max_df_thre
    #tf-idf feature extraction
    tfidf_vectorizer,tfidf_features,feature_names=tfidf_extractor(min_df_sh=min_df_threshold,
                                                                  max_df_sh=max_df_threshold,
                                                                  ngram_r=ngram_r,
                                                                  contents_list=cluster_contents_token_list)
    tfidf_features,feature_names=tfidf_del_element(tfidf_matrix=tfidf_features,
                                                   tfidf_names=feature_names,del_element='')
    #Topic modelin with NMF
    nmf_weights=nmf_topic_model_get(tfidf_features,total_topics=total_topics_num)
    nmf_topics=get_topics_terms_weights(nmf_weights,feature_names)
    nmf_topic_detail_list=output_topics(nmf_topics,total_topics=total_topics_num,
                                    display_weights=True,num_terms=topic_feature_display_num)
    #Get the context keywords of topic terms
    cluster_contents_total_token_list=list(Series(joined_contents_token_list)[Series(clusters)==str(cluster_index)])
    topics_keywords_context_tokens_size1=topics_keywords_context_tokens_get(topics=nmf_topic_detail_list,
                                                                            contents_token_list=cluster_contents_token_list,
                                                                            tfidf_feature_names=feature_names,
                                                                            if_joined=False,
                                                                            position=1)
    topics_keywords_context_tokens_size2=topics_keywords_context_tokens_get(topics=nmf_topic_detail_list,
                                                                            contents_token_list=cluster_contents_token_list,
                                                                            tfidf_feature_names=feature_names,
                                                                            if_joined=True,
                                                                            position=2)
    topics_keywords_context_tokens_size3=topics_keywords_context_tokens_get(topics=nmf_topic_detail_list,
                                                                            contents_token_list=cluster_contents_token_list,
                                                                            tfidf_feature_names=feature_names,
                                                                            if_joined=True,
                                                                            position=3)
    #Calculate the context weight
    topics_keywords_context_weights=topics_keywords_context_weights_get(context_list_size1=topics_keywords_context_tokens_size1,
                                        context_list_size2=topics_keywords_context_tokens_size2,
                                        context_list_size3=topics_keywords_context_tokens_size3,
                                        topics_tokens_names=feature_names,
                                        topics_tokens_scores=nmf_weights,
                                        position1_weight=1,position2_weight=0.9,
                                        position3_weight=0.8,
                                        context_word_count_weight=1,topic_weight=1)
    for topic_index in range(0,total_topics_num):#Save the topic terms and their context keywords of this cluster in the dataframe
        topic_keywords_num=len(topics_keywords_context_weights[topic_index])
        #If the number of topic terms in one theme is less than the displaying number of topic terms which is setted before, then make it up with "None".
        if topic_keywords_num<topic_feature_display_num:
            for add_index in range(0,topic_feature_display_num-topic_keywords_num):
                topics_keywords_context_weights[topic_index].append('None')
                nmf_topic_detail_list[topic_index].append(('None',0))
        cluster_topic_df[str(cluster_index+1)+str(topic_index)]=\
            [word for (word,weight) in nmf_topic_detail_list[topic_index]]
        cluster_topic_with_context[str(cluster_index+1)+str(topic_index)]=\
            topics_keywords_context_weights[topic_index]

#Generate the two layer column names of dataframe
clusters_name_columns=[]
topics_name_columns=[]
for cluster_index in range(1,clusters_num+1):
    for topic_index in range(1,total_topics_num+1):
        clusters_name_columns.append(str(cluster_index))
        topics_name_columns.append(chr(64+topic_index))
cluster_topic_with_context.columns=[clusters_name_columns,topics_name_columns]
cluster_topic_with_context.columns.names=['clusters','topics']
cluster_topic_df.columns=[clusters_name_columns,topics_name_columns]
cluster_topic_df.columns.names=['clusters','topics']

cluster_topic_with_context.to_csv('topics_context_'+file_name+'_'+
                                  topic_model_name+'.csv')
cluster_topic_df.to_csv(file_name+'_'+topic_model_name+'.csv')
