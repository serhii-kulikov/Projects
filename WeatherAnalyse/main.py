import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import class_file
from scipy import interpolate

#InputData
in_df = pd.read_csv ('weather.csv')
c_in_df = in_df.copy (deep = True)
day = pd.DataFrame({'Day': np.linspace (0, c_in_df.shape [0] - 1, c_in_df.shape [0]) + 1})
c_in_df = c_in_df.join (day)
CfPa = class_file.PlotAnalyse
CfBa = class_file.BarAnalyse

#MeanInformation at all of the parameters
#Temp
#df = c_in_df [['Day', 'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']]
#WindSpeed
#df = c_in_df [['Day', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']]
#Humidity
#df = c_in_df [['Day', 'Humidity9am', 'Humidity3pm']] 
#Pressure
#df = c_in_df [['Day', 'Pressure9am', 'Pressure3pm']]
#Cloud
#df = c_in_df [['Day', 'Cloud9am', 'Cloud3pm']]
#CfPa.Plot (CfPa.MeanData (df))

#Bar
#Tdf = c_in_df [['Day', 'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']]
#Mdf = c_in_df [['Day', 'Humidity9am', 'Humidity3pm']]
#CfBa.BarData (CfBa.PM_data (Tdf, Mdf))

#MeanYearInformation
def MeanInfo (c_in_df):
    #Temp
    Tdf = c_in_df [['Day', 'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']]
    Tdf = CfPa.MeanData (Tdf) [['Day', 'MeanData']].rename (columns = {'MeanData': 'MeanTemp'})
    #WindSpeed
    Wdf = c_in_df [['Day', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']]
    Wdf = CfPa.MeanData (Wdf) [['MeanData']].rename (columns = {'MeanData': 'MeanWindS'})
    #Humidity
    Hdf = c_in_df [['Day', 'Humidity9am', 'Humidity3pm']] 
    Hdf = CfPa.MeanData (Hdf) [['MeanData']].rename (columns = {'MeanData': 'MeanHum'})
    #Pressure
    Pdf = c_in_df [['Day', 'Pressure9am', 'Pressure3pm']]
    Pdf = CfPa.MeanData (Pdf) [['MeanData']].rename (columns = {'MeanData': 'MeanPress'})
    #Cloud
    Cdf = c_in_df [['Day', 'Cloud9am', 'Cloud3pm']]
    Cdf = CfPa.MeanData (Cdf) [['MeanData']].rename (columns = {'MeanData': 'MeanCloud'})
    MIdf = Tdf.join (Wdf)
    MIdf = MIdf.join (Hdf)
    MIdf = MIdf.join (Pdf)
    MIdf = MIdf.join (Cdf)
    print (MIdf [['MeanTemp', 'MeanWindS', 'MeanHum', 'MeanPress', 'MeanCloud']].describe ())
#MeanInfo (c_in_df)

#Mean/Max/Min Year Temperature
def Temp (c_in_df, in_df):
    Tdf = c_in_df [['Day', 'MinTemp', 'MaxTemp']]
    Tdf = CfPa.MeanData (Tdf) [['Day', 'MeanData']].rename (columns = {'MeanData': 'MeanTemp'})
    meanT = Tdf [['MeanTemp']].mean () ['MeanTemp']
    maxT = in_df [['MaxTemp']].max () ['MaxTemp']
    maxMT = Tdf [['MeanTemp']].max () ['MeanTemp']
    minT = in_df [['MinTemp']].min () ['MinTemp']
    minMT = Tdf [['MeanTemp']].min () ['MeanTemp']
    print ('Average year temperature = ', meanT, 'C')
    print ('Max year temperature = ', maxT, 'C')
    print ('The hottest month temperature = ', maxMT, 'C')
    print ('Min year temperature = ', minT, 'C')
    print ('The coolest month temperature = ', minMT, 'C')
#Temp (c_in_df, in_df)

def RainfallSum (in_df):
    c = 0
    for i in in_df ['Rainfall']:
        c += i
    print ('Rainfall Year = ', c, 'mm')
RainfallSum (in_df)


#'''
#Hot Summer mediterrian climate (Csa)  
#https://storymaps.arcgis.com/stories/345bc9c775d1424280b776ec38c3b1e1
#https://en.wikipedia.org/wiki/Mediterranean_climate
#'''
