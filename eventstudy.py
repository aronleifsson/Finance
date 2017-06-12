'''
@author: Aron Leifsson
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

###################
####Class Start####

class EventStudy:
    #Key assumptions: DFs for historical returns and factors follow the same timeseries and thus DF index is in datetime format
    def __init__(self, Events, Period, BurnIn, FamaFrenchFactors, HistoricalConstituentReturns, BackupFileName, IterStart=None, IterEnd=None):
        #Events:                       A list of Events. A DF with a date column ['Date'] and a symbol column ['Ticker'], 
        #                              indicating the date of event and the ticker symbol for the constituent.
        #
        #Period:                       a 1x2 np.array where the first column indicates the nr of days to be observed prior
        #                              to the event, and the second column indicates the nr of days to be observed post event.
        #
        #BurnIn:                       Integer indicating the nr of days used for estimating the
        #                              Fama French factors and intercept for the Five Factor Fama French model.
        #
        #FamaFrenchFactors:            A DF whose columns are the factors, and indices are dates.
        #HistoricalConstituentReturns: A DF whose columns are the constituents, and indices are dates.
        #
        #BackupFileName:               The name of the backup file for which all AR calculations are stored in.
        #
        #IterStart:                    Integer defining where to start in the index row nr in the event list
        
        self.__AR = np.array([None for i in range(0,Period[0]+Period[1]+1)]) #initialize array of arrays that contains the AR for each day for each event
        self.__AAR = None
        self.CAAR = None
        self.Events = Events
        self.Period = Period
        self.BurnIn = BurnIn
        self.__FamaFrenchFactors = FamaFrenchFactors
        self.__HistoricalConstituentReturns = HistoricalConstituentReturns
        self.BackupFileName = BackupFileName
        self.Reset = False # A variable that indicates whether a study has previously been conducted, thus signaling for a reset of variables AR, AAR and CAAR should another study be conducted with the same dataset but different periods and/or burnin.
        if IterStart == None: self.IterStart = 0
        else: self.IterStart = IterStart
        if IterEnd == None: self.IterEnd = Events.shape[0]
        else: self.IterEnd = IterEnd
        self.__BusinessDays = self.__FamaFrenchFactors.index.tolist()
    
    def __reset(self):
        self.__AR = np.array([None for i in range(0,self.Period[0]+self.Period[1]+1)]) #initialize array of arrays that contains the AR for each day for each event
        self.__AAR = None
        self.CAAR = None
        self.Reset = False
    
    def get_event_date(self,day): 
    #Definition: Find closest trading day after given event
    
    #Day:        Day of the event being analysed
    #(Note:      Be aware of the date format. We assume yyyy-mm-dd
    #            Also, be aware of the closing trading hour of the corresponding closing date)
    
        if day in self.__BusinessDays: return day
        else: 
            for i in range(1,30):
                next_day = day+pd.to_timedelta(i, unit="D")
                if next_day in self.__BusinessDays: return next_day
    
    def ConductStudy(self):
        
        if self.Reset == True: self.__reset()
        else: self.Reset = True
        
        with open(self.BackupFileName,'wb') as f_handle: # create a backup txt file that contains the iterations for each AR's of each event
            #Loop through each event in the EventList
            for i in tqdm(range(self.IterStart, self.IterEnd)):
                event_date = self.get_event_date(pd.to_datetime(self.Events.Date[i]))
                start = self.__BusinessDays[self.__BusinessDays.index(event_date)-(self.BurnIn+self.Period[0]+1)]
                end = self.__BusinessDays[self.__BusinessDays.index(event_date)+self.Period[1]]
                
                ConstituentReturn = self.__HistoricalConstituentReturns[[self.Events.Ticker[i]]].loc[start:end]
                FamaFrench = self.__FamaFrenchFactors.loc[start:end]
                
                # Calculate the response variable y (the excess returns)
                y = ConstituentReturn[self.Events.Ticker[i]].iloc[:self.BurnIn] - FamaFrench['RF'].iloc[:self.BurnIn]
                # Perform linear regression of y given the five Fama French factors
                lm = smf.ols(formula='y ~ Mkt_RF + SMB + HML + RMW + CMA', data=FamaFrench.iloc[:self.BurnIn]).fit()
                
                # Perform matrix operations to calculate the Expected return
                # Note: I need to add 1 at the end of lm.param to account for the RF column in FFF 
                #       when performing the matrix & vector multiplication
                #ExpectedReturn = FamaFrench.iloc[self.BurnIn+1:].dot(lm.params.iloc[1:].append(pd.Series([1],index=['RF'])))+lm.params[0]
                ExpectedReturn = FamaFrench['Mkt_RF'].iloc[self.BurnIn+1:]*lm.params[1]+FamaFrench['SMB'].iloc[self.BurnIn+1:]*lm.params[2]+FamaFrench['HML'].iloc[self.BurnIn+1:]*lm.params[3]+FamaFrench['RMW'].iloc[self.BurnIn+1:]*lm.params[4]+FamaFrench['CMA'].iloc[self.BurnIn+1:]*lm.params[5]+lm.params[0] + FamaFrench['RF'].iloc[self.BurnIn+1:]
                ExcessReturn = ConstituentReturn[self.Events.Ticker[i]].iloc[self.BurnIn+1:]-ExpectedReturn
                
                self.__AR = np.vstack((self.__AR,ExcessReturn))
                np.savetxt(f_handle,[ExcessReturn.values],delimiter='\t')
        
        self.__AAR = self.__AR[1:].mean(axis=0)
        self.CAAR = np.cumsum(self.__AAR)

    def plot(self):
        x=np.array([i+1 for i in range(-(self.Period[0]+1),self.Period[1])])
        plt.plot(x,self.CAAR)
        plt.axvline(x=0)
        plt.show()

####Class End####
#################

#Example Usage:

#Fetch list of positive/negative/neutral news events
EventsDF = pd.read_table('positive_news_events.txt')

#Fetch apropriate historical constituent prices:
HistoricalConstituentReturns = pd.read_csv('sp500_hist_mod_returns_v2.csv',index_col=0)
HistoricalConstituentReturns = HistoricalConstituentReturns.transpose()
HistoricalConstituentReturns.index = pd.to_datetime(HistoricalConstituentReturns.index)

#Fetch Fama French Factors:
FamaFrenchFactors = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv',index_col=0)
FamaFrenchFactors.index = pd.to_datetime(FamaFrenchFactors.index,format='%Y%m%d')

#Run the program
ES = EventStudy(EventsDF,np.array([60,60]),100,FamaFrenchFactors, HistoricalConstituentReturns, 'beckup_d.out',0,10000)
ES.ConductStudy()
ES.plot()