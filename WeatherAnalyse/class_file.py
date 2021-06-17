import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class BarAnalyse ():

    def PM_data (Tdf, Mdf):
        Tdf = PlotAnalyse.MeanData (Tdf) [['Day', 'MeanData']]
        Tdf = Tdf.rename (columns = {'MeanData': 'TempData'})
        Mdf = PlotAnalyse.MeanData (Mdf) [['MeanData']]  
        df = Tdf.join (Mdf)
    
        max_v = df ['MeanData'].max ()
        max_k = 0
        for k in df ['Day']:
            if df ['MeanData'] [k - 1] == max_v:
                max_k = df ['Day'] [k - 1]
        PMDict = {}
        for i in df ['Day']:
            if i % 30 == max_k % 30:
                PMDict [i] = df ['MeanData'] [i - 1]

        PMDict = pd.DataFrame (PMDict.items (), columns=['DayNumb', 'MeanData'])  
        PM_df = pd.DataFrame ({'Month':['May', 'June', 'July', 'August','September', 'October', 'November', 'December', 'January', 'February', 'March', 'April']})
        PM_df = PM_df.join (PMDict)
        return PM_df

    def BarData (PM_df):
        func = PM_df ['MeanData']
        arg = PM_df ['Month']
        fig, ax = plt.subplots ()    
        ax.bar (arg, func)
        fig.set_figwidth (12)
        fig.set_figheight (6)
        plt.show ()
    
class PlotAnalyse ():
    
    def MeanData (df):
        cdf = df.copy (deep = True)
        mean = df ['Day'] - df ['Day']
        del cdf ['Day']
        for j in cdf:
            mean += cdf [j]
        mean /= cdf.shape [1]    
        tck = interpolate.splrep (df ['Day'], mean, k = 5, s = 5000)
        y_tck = interpolate.splev (df ['Day'], tck, der = 0)
        mean_df = pd.DataFrame ({'MeanData': mean, 'IntrpMeanData': y_tck })
        new_df = df.join (mean_df)
        return new_df
    
    def Plot (df):
        cdf = df.copy (deep = True)
        del cdf ['Day']
        for i in cdf:
            if i == 'IntrpMeanData':
                plt.plot (df ['Day'], cdf [i], color = 'red',  label = i, lw = '1.5')
            elif i == 'MeanData':
                plt.plot (df ['Day'], cdf [i], color = 'black',  label = i, lw = '1')
            else:
                plt.plot (df ['Day'], cdf [i], label = i, lw = '0.5')
        plt.legend ()
        plt.show ()  
