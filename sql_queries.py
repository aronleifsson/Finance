'''
Created on Jun 13, 2017

@author: Aron
'''
import pypyodbc
import pandas as pd
import numpy as np

def get_market_date(day, BusinessDays): 
#Definition: Find closest trading day after given event

#Day:        Day of the event being analysed of type Timestamp
#(Note:      Be aware of the date format. We assume yyyy-mm-dd
#            Also, be aware of the closing trading hour of the corresponding closing date)
    
    if day.replace(hour=0, minute=0, second=0) in BusinessDays and day.hour < 16: #assuming the stock exchange closes at 16:00 
        return day.replace(hour=0, minute=0, second=0) #we use .replace since BusinessDays is of form timestamp yyyy-mm-dd hh:mm:ss where hh=mm=ss=0
    else: 
        for i in range(1,30):
            next_day = day.replace(hour=0, minute=0, second=0)+pd.to_timedelta(i, unit="D")
            if next_day in BusinessDays: return next_day

def connect():
    return pypyodbc.connect('Driver={SQL Server};'
                            'Server=ARON-INSP15\SQLEXPRESS;'
                            'Database=AAM;'
                            'Trusted_connection=yes') 


def get_events(Sentiment,Year_Start, Year_End):
    '''
    Select all extremely positive/neutral/negative news events.
    Constraints: We do not want more than one news story per constituent per day
    Assumption: The first time this event on a given day occurs is stored, if another event for the same constituent on the same
                day occurs, we drop it, keeping the timestamp for the first/initial event that day for that constituent.
    
    This is tricky to do in pure SQL, so we use pandas in addition to SQL
    '''
    if Sentiment == 1: condition = '[Prob_POS]>0.75 AND [Sentiment]=1'
    elif Sentiment == 0: condition = '[Prob_NTR]>0.75 AND [Sentiment]=0'
    else: condition = '[Prob_NEG]>0.75 AND [Sentiment]=-1'
    connection = connect()
    SQLCommand = ("SELECT DISTINCT [Timestamp] as Date"
                  ",[Ticker]"
                  "FROM [AAM].[dbo].[sp500_sentiment]"
                  "WHERE [Relevance]=1 AND [Novelty]>=5 AND "+condition+" AND YEAR([Timestamp])>="+str(Year_Start)+" AND YEAR([Timestamp])<= "+str(Year_End)+" AND [Source]='DJN'"
                  "ORDER BY Date ASC") 
    
    df = pd.read_sql(SQLCommand, connection,index_col='date')
    connection.close()
    
    del df.index.name
    df.index = pd.to_datetime(df.index) #set index as the timestamps for each event
    
    df['Date'] = pd.Series(df.index.date, index=df.index) #create date column that only displays the day-month-year of the event
    df = df.drop_duplicates() #drop duplicates, thus keeping the first instance of the event that day
    del df['Date'] #delete the date column
    return df.rename(columns={'ticker':'Ticker'})


def get_daily_aggr_sent_score(Year_Start, Year_End):
    '''
    Naive approach to calculating the daily aggregate sentiment score, but for the purposes of this project it will do.
    '''
    connection = connect()
    SQLCommand_Bdays = ("SELECT * FROM [AAM].[dbo].[bdays] WHERE YEAR([Date]) >="+str(Year_Start)+" AND YEAR([Date]) <="+str(Year_End+1))
    BDays = pd.read_sql(SQLCommand_Bdays, connection)
    BDays = pd.to_datetime(BDays['date']).tolist()
    
    SQLCommand = ("SELECT [Timestamp],[Sentiment]"
                  "FROM [AAM].[dbo].[sp500_sentiment]"
                  "WHERE [Relevance] >=0.5"
                  "AND [Sentiment] != 0"
                  "AND YEAR([Timestamp]) >="+str(Year_Start)+" AND YEAR([Timestamp]) <="+str(Year_End)+
                  "ORDER BY [Timestamp] ASC")
    SentimentDF = pd.read_sql(SQLCommand, connection)
    
    SentimentDF['timestamp'] = pd.to_datetime(SentimentDF['timestamp'])
    SentimentDF['MarketDate'] = SentimentDF['timestamp'].apply(lambda x: get_market_date(x,BDays))
    del SentimentDF['timestamp']
    count = SentimentDF.groupby(['MarketDate', 'sentiment']).size().unstack().fillna(0)
    del count.index.name
    count['S_t'] = np.log10((count[1]+1)/(count[-1]+1))
    count.to_csv('sp500_aggr_sent_score_2000_2016.csv')
    return count

#print(get_daily_aggr_sent_score(2000,2017))