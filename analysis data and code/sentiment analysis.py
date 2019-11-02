import jieba
import jieba.analyse
import pandas as pd
from pandas import DataFrame
from pandas import Series
import csv
import time

#Upload content from file
def get_contents(file_name):
    contents_file=pd.read_csv(open(file_name+'.csv',encoding='utf-8'))
    contents_list=[]
    time_list=[]
    for i in range(len(contents_file.index)):
        contents_list.append(contents_file.ix[i]['ans_con'].strip())
        time_list.append(contents_file.ix[i]['time'].strip())
    return contents_list,time_list,contents_file

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

#Transform the time format
def time_format_transform(raw_time_list,time_granularity):
    transformed_time_list=[]
    for time_str in raw_time_list:
        transformed_time=time.strptime(time_str,"%Y-%m-%d %H:%M:%S")
        if time_granularity=='month':
            transformed_time_list.append(time.strftime("%Y-%m",transformed_time))
        elif time_granularity=='year':
            transformed_time_list.append(time.strftime("%Y",transformed_time))
    return transformed_time_list

#Parameters of sentiment analysis
token_calculate_range=6
noword_weaken_degree=0.25
#Text segmentation and cleaning
jieba.load_userdict('segmentation.txt')
contents_list,time_list,contents_file=get_contents('Zhihu answer contents')
contents_token_list=get_contents_token_list(contents_list)
contents_token_list=del_blank_character(contents_token_list)
#Upload lexicons
sentiment_words_dict=get_sentiment_words('sentiment score.txt')
degree_words_dict=get_degree('degree.csv')
nowords_list=get_words('nowords.txt')
#Detect the sentiment, degree and negative words
detect_words_list=detect_words_list_get(contents_token_list,sentiment_words_dict,
                          degree_words_dict,nowords_list)
#Calculate the sentiment score of each content
contents_sentiment_score_list=calculate_sentiment_score(detect_words_list,
                                                        calculate_range=token_calculate_range,
                                                        noword_weaken=noword_weaken_degree)

contents_sentiment_score_df=DataFrame(contents_sentiment_score_list,
                                      columns=['positive','negative','final'])
contents_sentiment_score_df['time']=time_list
contents_sentiment_score_df.to_csv('sentiment analysis results.csv',index=False)
#Transform the time format
contents_sentiment_score_df['time']=time_format_transform(time_list,'month')
contents_sentiment_score_df.to_csv('sentiment analysis results with month time format.csv',index=False)
contents_sentiment_score_df['time']=time_format_transform(time_list,'year')
contents_sentiment_score_df.to_csv('sentiment analysis results with year time format.csv',index=False)