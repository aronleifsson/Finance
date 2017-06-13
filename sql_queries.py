'''
@author: Aron
'''
import pypyodbc
import pandas as pd

def connect():
    return pypyodbc.connect('Driver={SQL Server};'
                            'Server=ARON-INSP15\SQLEXPRESS;'
                            'Database=AAM;'
                            'Trusted_connection=yes') 


def get_events(Sentiment,Year_Start, Year_End):
    #sentiment, Year_Start and Year_End are all integers. Sentiment \in [-1,1]
    '''
    Select all extremely positive/neutral/negative news events.
    Constraints: We do not want more than one news story per constituent per day
    Assumption: The first time this event on a given day occurs is stored, if another event for the same constituent on the same
                day occurs, we drop it, keeping the timestamp for the first/initial event that day for that constituent.
    
    This is tricky to do in pure SQL, so we use pandas in addition to SQL
    '''
    if Sentiment == 1: condition = '[Prob_POS]>0.99 AND [Sentiment]=1'
    elif Sentiment == 0: condition = '[Prob_NTR]>0.99 AND [Sentiment]=0'
    else: condition = '[Prob_NEG]>0.99 AND [Sentiment]=-1'
    connection = connect()
    SQLCommand = ("SELECT DISTINCT [Timestamp] as Date"
                  ",[Ticker]"
                  "FROM [AAM].[dbo].[sp500_sentiment]"
                  "WHERE [Relevance]=1 AND [Novelty]=1 AND "+condition+" AND [Confidence] >0.98 AND YEAR([Timestamp])>="+str(Year_Start)+" AND YEAR([Timestamp])<= "+str(Year_End)+" AND [Source]='DJN'"
                  "ORDER BY Date ASC") 
    
    df = pd.read_sql(SQLCommand, connection,index_col='date')
    connection.close()
    
    del df.index.name
    df.index = pd.to_datetime(df.index) #set index as the timestamps for each event
    
    df['Date'] = pd.Series(df.index.date, index=df.index) #create date column that only displays the day-month-year of the event
    df = df.drop_duplicates() #drop duplicates, thus keeping the first instance of the event that day
    del df['Date'] #delete the date column
    return df.rename(columns={'ticker':'Ticker'})

