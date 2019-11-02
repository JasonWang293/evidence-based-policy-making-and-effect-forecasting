import jieba
import jieba.analyse
import pandas as pd
from pandas import DataFrame
import numpy as np
from scipy.sparse import csr_matrix
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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
def tfidf_extractor(contents_list,min_df_sh=1,max_df_sh=1.0,ngram_r=(1,1)):
    vectorizer=TfidfVectorizer(analyzer=lambda x:x.split(' '),
                               min_df=min_df_sh,
                               norm='l2',
                               smooth_idf=True,
                               use_idf=True,
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

#Get text cluster with K-means
def k_means_cluster_get(feature_matrix,num_clusters=5,max_iter_num=10000):
    km=KMeans(n_clusters=num_clusters,max_iter=max_iter_num)
    km.fit(feature_matrix)
    clusters=km.labels_
    return km,clusters

#Display details of text clusters
def cluster_detail_get(cluster_obj,contents_dataframe,feature_names,num_clusters=5,
                     top_features_num=10,file_name=None,tfidf_matrix=None):
    cluster_details=pd.DataFrame(columns=['key_features','contents','contents_num'],index=range(num_clusters))
    #Get sorted cluster centers
    ordered_centroids=cluster_obj.cluster_centers_.argsort()[:,::-1]
    #Get key features of each cluster
    for cluster_num in range(num_clusters):
        #Get the name of each feature
        key_features=[(feature_names[index],
                       round(cluster_obj.cluster_centers_[cluster_num][index],2))
                      for index in ordered_centroids[cluster_num,:top_features_num]]
        cluster_details.ix[cluster_num]['key_features']=key_features
        contents=contents_dataframe[contents_dataframe['Cluster']==cluster_num]['ans_num'].values.tolist()
        cluster_details.ix[cluster_num]['contents']=contents
        cluster_details.ix[cluster_num]['contents_num']=len(contents)
    if file_name:
        cluster_details.to_csv(file_name+'.csv')
    return cluster_details

#Save results in txt file
def save_txt_file_whole(data_list,filename):
    data_file=open(filename+'.txt','w+',encoding='utf-8')
    data_file.write(data_list)
    data_file.close()

#Save text cluster results in csv file
def save_by_cluster(contents_df,file_name,cluster_index=None):
    if cluster_index==None:
        contents_df.to_csv(file_name+'.csv',index=None)
    else:
        df=contents_df[contents_df['Cluster']==cluster_index]
        df.to_csv(file_name+'.csv',index=None)

#Text segmentation and cleaning
jieba.load_userdict('segmentation.txt')
contents_list,contents_file=get_contents('Zhihu answer contents')
contents_nolabel_list=del_html_label(contents_list)
contents_token_list=get_contents_token_list(contents_nolabel_list)
contents_token_list=del_blank_character(contents_token_list)
stopwords_list=get_stopwords('stopwords.txt')
contents_token_cleaned_list=del_stopwords(contents_token_list,stopwords_list)
joined_contents_token_cleaned_list=get_joined_content_tokens(contents_token_cleaned_list)
#Text clustering parameters
num_clusters=18
min_df_threshold=2
max_df_threshold=0.9
ngram_r=(1,1)
top_features_show_num=15
#tf-idf feature extraction
tfidf_vectorizer,tfidf_features,feature_names=tfidf_extractor(min_df_sh=min_df_threshold,max_df_sh=max_df_threshold,
                                                              contents_list=joined_contents_token_cleaned_list,
                                                              ngram_r=ngram_r)
tfidf_features,feature_names=tfidf_del_element(tfidf_matrix=tfidf_features,
                                               tfidf_names=feature_names,del_element='')
'''tfidf_df=display_features(features=np.round(tfidf_features.todense(),2),
                          feature_names=feature_names,file_name='tfidf')'''
#Text clustering with K-means
km_obj,km_clusters=k_means_cluster_get(feature_matrix=tfidf_features,
                                       num_clusters=num_clusters)
km_clusters=list(km_clusters)
#tf-idf of each centroid
clusters_center_df=DataFrame(km_obj.cluster_centers_,columns=feature_names)
clusters_center_df.to_csv('tf-idf of each centroid.csv',index=None)
contents_file['Cluster']=km_clusters
save_txt_file_whole(str(km_clusters),'km_clusters_list')
#The number of contents in each cluster
km_clusters_count=Counter(km_clusters)
print(km_clusters_count)
km_cluster_detail=cluster_detail_get(km_obj,contents_file,feature_names,num_clusters=num_clusters,
                                     top_features_num=top_features_show_num,
                                     file_name='km_cluster_detail')
'''save_by_cluster(contents_df=contents_file,
                file_name='text clustering results',
                cluster_index=1)'''
save_by_cluster(contents_df=contents_file,
                file_name='text clustering results')