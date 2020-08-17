import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import seaborn as sns
from scipy.stats import chisquare, norm, chi2_contingency

"""
# Save a particular sheet or dataframe of excel as a separate csv file
dataset = pd.read_excel("DataDrop20200703.xlsx", sheet_name='Sheet3')
# Remove rows with RegionRes not at NCR
# dataNEW = dataset[dataset.RegionRes == 'NCR']
pd.DataFrame.to_csv(dataset,"DataDrop20200703.csv", index=False)
"""

dataset = pd.read_csv('DataDropNCR20200703.csv', parse_dates=['DateRepConf','DateDied'])

"""
#check info of dataset
print(dataset.info())
#check column data types
print(dataset.dtypes)
#determine count, mean, stdev etc.
print(dataset.describe())
"""       

#plot data with date:
def plotting(dataset,sort_column,plot,labely,title,savename):
    #sort dataset by specific column
    dataset = dataset.sort_values(by=sort_column)
    data = dataset[sort_column].dropna()
    #count frequency of occurrence of each data (k as datetime object)
    data_count = Counter(data)
    #change datetime into string with specified format
    data_objects = [k.strftime('%b %d') for k,v in data_count.items()]
    
    #determine quarantine status from dates
    quarantine, count, total = list(), list(), list()
    for k, v in data_count.items():
        if (k.month <= 3) and (k.day in range(1,15)):
            quarantine.append('Before Quarantine')
        elif (k.month==5) and (k.day in range(16,32)):
            quarantine.append('MECQ')
        elif (k.month >= 6) and (k.day in range(1,32)):
            quarantine.append('GCQ')
        else:
            quarantine.append('ECQ')
        count.append(v)

    #determine total cases for each day
    prev = 0
    for i in range(len(count)):
        prev += count[i]
        total.append(prev)

    #create dataframe with date, case counts and quarantine status
    data_dict = {'Date Confirmed': data_objects, 'Number of Cases': total,\
					'Quarantine Status': quarantine}
    data_frame = pd.DataFrame(data_dict, index=data_count.keys(),\
			columns=['Date Confirmed', 'Number of Cases','Quarantine Status'])

    sns.set_style("darkgrid", {'xtick.bottom': True, 'axes.spines.right': False,\
					'axes.spines.top': False})
    plt.figure(figsize=(15,5))
    
    #choose between a barplot or scatterplot
    if plot == 'bar':
        g = sns.barplot(data=data_frame, x='Date Confirmed', y='Number of Cases',\
				hue='Quarantine Status', hue_order=['Before Quarantine','ECQ','MECQ',\
					'GCQ'], palette="CMRmap", dodge=False) 
        plt.legend(loc='upper left', title='Quarantine Status')
        g.set_xticklabels(g.get_xticklabels(), rotation=70,\
							horizontalalignment='center', fontsize=6)     
        g.set(xlabel=None, ylabel=labely, title=title) 
		
    elif plot == 'scatter':
        g = sns.stripplot(data=data_frame, x='Date Confirmed', y='Number of Cases',\
							hue='Quarantine Status', hue_order=['Before Quarantine',\
								'ECQ','MECQ','GCQ'], palette="CMRmap", dodge=False)
        plt.legend(loc='upper left', title='Quarantine Status')
        g.set_xticklabels(g.get_xticklabels(), rotation=70,\
							horizontalalignment='center', fontsize=6)     
        g.set(xlabel=None, ylabel=labely, title=title) 
		
    else:
       None 
       
    # plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
    plt.savefig(savename, bbox_layout='tight',dpi=300)
    plt.show()

########################

#plotting for data without date (histogram, countplot):
def plot_nodate(dataset,sort_column,plot,title,labelx,labely,savename):
    sns.set_style("darkgrid", {'xtick.bottom': True, 'axes.spines.right': False,\
								'axes.spines.top': False})
    plt.figure(figsize=(10,5))

    #plotting
    if plot=='histogram':
        #sort dataset by specific column
        dataset = dataset.sort_values(by=sort_column)
        data = dataset[sort_column].dropna()

        mean, stdev = norm.fit(data)  #get mean and standard deviation of data
        label = "normal distribution fit:\nmean={:.5f},stdev={:.5f}"\
						.format(mean,stdev)
        sns.distplot(data, kde=False, bins=25, fit=norm,\
						fit_kws={'color':'red','label':label})
        plt.legend(loc='upper right')

    elif plot=='categorical':
        #sort dataset by specific column
        dataset = dataset.sort_values(by='HealthStatus')
        data = dataset[sort_column].dropna()

        p = sns.countplot(x=sort_column,data=dataset,hue='HealthStatus')
        p.legend(loc='upper left')
        leg = p.get_legend()
        leg.set_title("Health Status")
        labs = leg.texts
        label = ['Asymptomatic','Critical','Died','Mild','Recovered','Severe']
        for i in range(len(label)):
            labs[i].set_text(label[i])
    
    else:
        None

    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)

    plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
    plt.savefig(savename, dpi=300)
    plt.show()

###########################

#determine chi-square using scipy function
def scipy_chi(data):

    data = data.dropna()
    if data.dtype == 'datetime64[ns]':
        data_ord = data.map(datetime.toordinal)
    else:
        data_ord = data
    data_ord = data_ord.values.reshape(-1,1)

    return print(chisquare(data_ord))

##########################

#create contingency table for sex and healthstatus
def contingency(dataset,kind):

	#prints observed values
    if kind=='observed':
        df2 = pd.crosstab(dataset['Sex'],dataset['HealthStatus'],margins=True)
        df2.columns = ['Asymptomatic','Critical','Died','Mild','Recovered','Severe','TOTAL']
        df2.index = ['Female','Male','TOTAL']
        print(df2)
	
	#prints expected values
    elif kind=='expected':
        df2 = pd.crosstab(dataset['Sex'],dataset['HealthStatus'],margins=False)
        chi, p, dof, arr = chi2_contingency(df2)
        df2_exp = pd.DataFrame({'Asymptomatic': arr[:,0], 'Critical': arr[:,1],\
					'Died': arr[:,2], 'Mild': arr[:,3], 'Recovered': arr[:,4],\
						'Severe': arr[:,5]}, index=['Female', 'Male'])
        print('{}\nchi-square={}, p-value={}, dof={}'.format(df2_exp,chi,p,dof))

#plotting
plotting(dataset,'DateRepConf','bar','ylabel','title','NumCasesBarPlot.png')