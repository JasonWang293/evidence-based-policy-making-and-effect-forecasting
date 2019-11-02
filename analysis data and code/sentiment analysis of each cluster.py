import jieba
import jieba.analyse
import pandas as pd
from pandas import DataFrame
from pandas import Series
import csv

#Upload content from file
def get_contents(file_name):
    contents_file=pd.read_csv(open(file_name+'.csv',encoding='utf-8'))
    contents_list=[]
    for i in range(len(contents_file.index)):
        contents_list.append(contents_file.ix[i]['ans_con'].strip())
    return contents_list,contents_file

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

#Upload words from txt file
def get_words(file_name):
    words=[str(line.encode('utf-8').decode('utf-8-sig').strip()) for line in open(file_name,'r',encoding='utf-8')]
    return words

#Upload the sentiment lexicon
def get_sentiment_words(file_name):
    sentiment_words_file=[str(line.encode('utf-8').decode('utf-8-sig').strip()) for line in open(file_name,'r',encoding='utf-8')]
    sentiment_words={}
    for i in sentiment_words_file:
        if i:
            sentiment_words[i.split(' ')[0]]=float(i.split(' ')[1])
    return sentiment_words

#Upload the degree adverb lexicon
def get_degree(file_name):
    degree={}
    degree_file=csv.reader(open(file_name,'r',encoding='utf-8'))
    for line in degree_file:
        #print(line[0].encode('utf-8').decode('utf-8-sig').strip())
        #print(line[1].encode('utf-8').decode('utf-8-sig').strip())
        degree[line[0].encode('utf-8').decode('utf-8-sig').strip()]=float(
                line[1].encode('utf-8').decode('utf-8-sig').strip())
    return degree

#Detect degree words, sentiment words and negative words in contents, and record their position, category and score in list.
def detect_words_list_get(contents_token_list,sentiment_words_dict,
                          degree_words_dict,nowords_list):
    detect_words_list=[]
    content_num=0
    for content in contents_token_list:
        detect_words_list.append([])
        for i in range(len(content)):
            if content[i] in degree_words_dict:
                detect_words_list[content_num].append([i,'degree',
                                                       degree_words_dict[content[i]]])
            elif content[i] in sentiment_words_dict:
                detect_words_list[content_num].append([i,'sentiment',
                                                       sentiment_words_dict[content[i]]])
            elif content[i] in nowords_list:
                detect_words_list[content_num].append([i,'noword',
                                                       -1])
        content_num=content_num+1
    return detect_words_list

#Calculate the sentiment score of each content, including the positive, negative and final score.
def calculate_sentiment_score(detect_words_list,calculate_range=6,noword_weaken=0.25):
    contents_sentiment_score_list=[]
    content_num=0
    for content in detect_words_list:
        positive_score=0
        negative_score=0
        total_score=0
        for word_index in range(len(content)):
            if content[word_index][1]=='sentiment':
                word_score=float(content[word_index][2])
                nowords_num=0#Record the number of negative words
                #Detect the degree words and negative words in the context of sentiment words
                for back in range(1,calculate_range+1):
                    if content[word_index-back][1]=='sentiment':
                        break
                    elif (content[word_index][0]-content[word_index-back][0])>calculate_range:
                        break
                    elif content[word_index-back][1]=='degree':
                        word_score=word_score*float(content[word_index-back][2])
                        if content[word_index-back-1][1]=='noword':
                            word_score=word_score*noword_weaken#If there is a degree adverb between the negative word and sentiment word, weaken the sentiment score.
                    elif content[word_index-back][1]=='noword':
                        nowords_num=nowords_num+1
                #The influence of negative words
                if nowords_num>0:
                    word_score=((-1)**nowords_num)*word_score
                if word_score>0:
                    positive_score=positive_score+word_score
                else:
                    negative_score=negative_score+word_score
        total_score=positive_score+negative_score
        contents_sentiment_score_list.append([positive_score,negative_score,total_score])
        content_num=content_num+1
    return contents_sentiment_score_list

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
contents_token_list=get_contents_token_list(contents_list)
contents_token_list=del_blank_character(contents_token_list)
sentiment_words_dict=get_sentiment_words('sentiment score.txt')
degree_words_dict=get_degree('degree.csv')
nowords_list=get_words('nowords.txt')
#Parameters of sentiment analysis
clusters_num=18
token_calculate_range=6
noword_weaken_degree=0.25
min_df_thre=2
clusters_sentiment_df=pd.DataFrame(columns=range(1,clusters_num+1))#Save the sentiment analysis results of each cluster
#Upload the cluster index of each content
file_name='km_clusters'
clusters=read_txt_file_whole(file_name+'_list')
#Calculate the sentiment score of each content
detect_words_list=detect_words_list_get(contents_token_list,sentiment_words_dict,
                          degree_words_dict,nowords_list)
contents_sentiment_score_list=calculate_sentiment_score(detect_words_list,
                                                        calculate_range=token_calculate_range,
                                                        noword_weaken=noword_weaken_degree)
cluster_describe_df=DataFrame()#Record the descriptive statistics of the sentiment analysis results in each cluster
cluster_plot_score_list=[]#Record the descriptive statistics of each cluster that used to plot, including the sum, average and median of positive, negative and final sentiment score
#Calculate the sentiment analysis score of each cluster
for cluster_index in range(0,clusters_num):
    #Get the sentiment score of contents in the cluster
    cluster_contents_score=DataFrame(contents_sentiment_score_list)[pd.Series(clusters)==str(cluster_index)]
    #Calculate the sum of sentiment score of contents in the cluster
    cluster_sum=DataFrame(cluster_contents_score.sum(axis=0)).T
    cluster_sum.index=['sum']
    #Get the descriptive statistics of sentiment score of contents in the cluster
    cluster_describe=cluster_contents_score.describe()
    cluster_describe=cluster_sum.append(cluster_describe)
    cluster_describe_df=pd.concat([cluster_describe_df,cluster_describe],axis=1)
    cluster_plot_score=list(cluster_describe.ix['sum'])+list(cluster_describe.ix['mean'])+\
                       list(cluster_describe.ix['50%'])
    cluster_plot_score_list.append(cluster_plot_score)

#Record the descriptive statistics of sentiment score of all contents
total=DataFrame(contents_sentiment_score_list)
total_sum=DataFrame(total.sum(axis=0)).T
total_sum.index=['sum']
total_describe=total.describe()
total_score=total_sum.append(total_describe)
cluster_describe_df=pd.concat([cluster_describe_df,total_score],axis=1)

#Generate the two layer column names of dataframe
clusters_name_columns=[]
polarity_name_columns=['positive','negative','final']*(clusters_num+1)
for cluster_index in range(1,clusters_num+1):
    for i in range(3):
        clusters_name_columns.append(str(cluster_index))
clusters_name_columns=clusters_name_columns+['total']*3
cluster_describe_df.columns=[clusters_name_columns,polarity_name_columns]
cluster_describe_df.columns.names=['clusters','polarity']

cluster_describe_df.to_csv('sentiment_'+file_name+'.csv')

#Record the descriptive statistics of each cluster that used to plot in dataframe
parameter_name_columns=['sum']*3+['mean']*3+['median']*3
polarity_name_columns=['positive','negative','final']*3
cluster_plot_score_df=DataFrame(cluster_plot_score_list)
cluster_plot_score_df.columns=[parameter_name_columns,polarity_name_columns]
cluster_plot_score_df.columns.names=['parameter','polarity']
cluster_plot_score_df.to_csv('sentiment_plot_'+file_name+'.csv')

