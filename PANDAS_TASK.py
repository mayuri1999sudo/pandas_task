#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1.Import pandas under the name `pd`
import pandas as pd


# In[4]:


#2.Print the version of pandas that has been imported
pd.__version__


# In[5]:


#3.Print out all the version information of the libraries that are required by the pandas library
pd.show_versions()


# In[5]:


#4.Create a DataFrame `df` from this dictionary `data` which has the index `labels`
import numpy as np
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df=pd.DataFrame(data,index=labels)


# In[10]:


df


# In[11]:


#5.Display a summary of the basic information about this DataFrame and its data
print(df.info())


# In[12]:


#6. Return the first 3 rows of the DataFrame `df`
df.head(3)


# In[13]:


#7.Select just the 'animal' and 'age' columns from the DataFrame `df`
my_columns=["animal","age"]
df[my_columns]


# In[15]:


#8.Select the data in rows `[3, 4, 8]` and in columns `['animal', 'age']`
my_columns=["animal","age"]
df[my_columns]
df.loc[["d","e","i"]]


# In[20]:


#9.Select only the rows where the number of visits is greater than 3
df[df['visits']>3]


# In[31]:


#10.Select the rows where the age is missing, i.e. is `NaN`
df[df['age'].isnull()]


# In[29]:


#11.Select the rows where the animal is a cat *and* the age is less than 3
df[(df['animal']=='cat') & (df['age']<3)]


# In[38]:


#12.Select the rows the age is between 2 and 4 (inclusive)
df[(df['age']<=4)&(df['age']>=2)]


# In[41]:


#13.Change the age in row 'f' to 1.5
df.loc['f', 'age'] = 1.5
print(df)


# In[43]:


#14.Calculate the sum of all visits (the total number of visits)
df['visits'].sum()


# In[44]:


#15.Calculate the mean age for each different animal in `df`
df['visits'].mean()


# In[45]:


#16. Append a new row 'k' to `df` with your choice of values for each column. Then delete that row to return the original DataFrame.
df.loc['k'] = ['cat', '3', '2', 'no']
print("Print all records after insert a new row:")
print(df)


# In[46]:


print("\nDelete the new row and display the original  rows:")
df = df.drop('k')
print(df)


# In[49]:


#17.Count the number of each type of animal in `df`
df.groupby('animal').count()


# In[52]:


#18.Sort `df` first by the values in the 'age' in *decending* order, then by the value in the 'visit' column in *ascending* order
df.sort_values('age', ascending=False)


# In[54]:


df.sort_values('visits', ascending=True)


# In[55]:


#19.The 'priority' column contains the values 'yes' and 'no'. Replace this column with a column of boolean values: 'yes' should be `True` and 'no' should be `False`
df['priority'] = df['priority'].map(
                   {'yes':True ,'no':False})
df


# In[56]:


df = df.replace({'priority': {'yes': True, 
                                'no': False}})
df
  


# In[57]:


#20.In the 'animal' column, change the 'snake' entries to 'python'
df = df.replace({'animal': {'snake': 'python'}})
df


# In[6]:


#21. For each animal type and each number of visits, find the mean age. In other words, each row is an animal, each column is a number of visits and the values are the mean ages (hint: use a pivot table).
 mean=df.pivot_table(index="animal", columns="visits", aggfunc="mean")["age"]


# In[9]:


#22. You have a DataFrame df with a column 'A' of integers. For example:
#df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
#How do you filter out rows which contain the same integer as the row immediately above?
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
df.loc[df['A'].shift() != df['A']]


# In[10]:


#23. Given a DataFrame of numeric values, say
#df = pd.DataFrame(np.random.random(size=(5, 3))) # a 5x3 frame of float values
#how do you subtract the row mean from each element in the row?

df = pd.DataFrame(np.random.random(size=(5, 3)))
print(df)
print("\n")
df.sub(df.mean(axis=1), axis=0)


# In[11]:


#24. Suppose you have DataFrame with 10 columns of real numbers, for example:
#df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
#Which column of numbers has the smallest sum? (Find that column's label.)

df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
print(df)
df.sum(axis=0)
print("Column name:",list(dict(df.sum()).keys())[list(dict(df.sum()).values()).index(min(df.sum()))],",","sum value:",min(df.sum())) 

#or

#Another example
df=pd.DataFrame({'a':[1,2,3,4,5,6,],'b':[3,5,6,7,8,5],'c':[1,4,5,7,8,4],'d':[6,7,5,4,3,2],'e':[32,3,5,2,3,2],'f':[1,5,4,3,6,7]})
print(df)
df.sum(axis=0)
print("Column name:",list(dict(df.sum()).keys())[list(dict(df.sum()).values()).index(min(df.sum()))],",","sum value:",min(df.sum())) 


# In[12]:


#25. How do you count how many unique rows a DataFrame has (i.e. ignore all rows that are duplicates)?

data = { 
        'node1': [2,2,2,5,3,5,5],
        'node2': [6,6,8,77,25,10,77],
        'weight': [1,1,1,2,1,1,2], }
labels =  ['a','b','c','d','e','f','g']       
df = pd.DataFrame(data, columns = ['node1','node2','weight'],index = labels )
len(df.drop_duplicates(keep=False))


# In[13]:


#26. You have a DataFrame that consists of 10 columns of floating--point numbers. Suppose that exactly 5 entries in each row are NaN values. For each row of the DataFrame, find the column which contains the third NaN value.
#(You should return a Series of column labels.)

import numpy as np
nan = np.nan

data = [[0.04,  nan,  nan, 0.25,  nan, 0.43, 0.71, 0.51,  nan,  nan],
        [ nan,  nan,  nan, 0.04, 0.76,  nan,  nan, 0.67, 0.76, 0.16],
        [ nan,  nan, 0.5 ,  nan, 0.31, 0.4 ,  nan,  nan, 0.24, 0.01],
        [0.49,  nan,  nan, 0.62, 0.73, 0.26, 0.85,  nan,  nan,  nan],
        [ nan,  nan, 0.41,  nan, 0.05,  nan, 0.61,  nan, 0.48, 0.68]]

columns = list('abcdefghij')

df = pd.DataFrame(data, columns=columns)
print(df)
(df.isnull().cumsum(axis=1) == 3).idxmax(axis=1)


# In[14]:


#27. A DataFrame has a column of groups 'grps' and and column of numbers 'vals'. For example:
#df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'),'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})
#For each group, find the sum of the three greatest values.

df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})
print(df)
df.groupby('grps')['vals'].nlargest(3).sum(level=0)


# In[15]:


#28. A DataFrame has two integer columns 'A' and 'B'. The values in 'A' are between 1 and 100 (inclusive). For each group of 10 consecutive integers in 'A' (i.e. (0, 10], (10, 20], ...), calculate the sum of the corresponding values in column 'B'.

import numpy as np
df = pd.DataFrame(np.random.RandomState(8765).randint(1, 101, size=(100, 2)), columns = ["A", "B"])

print(df)
df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()


# In[16]:


#29. Consider a DataFrame df where there is an integer column 'X':
#df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})
#For each value, count the difference back to the previous zero (or the start of the Series, whichever is closer). These values should therefore be [1, 2, 0, 1, 2, 3, 4, 0, 1, 2]. Make this a new column 'Y'.

df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})
print(df)
df['Y'] = df.groupby((df['X'] == 0).cumsum()).cumcount()
# We're off by one before we reach the first zero.
first_zero_idx = (df['X'] == 0).idxmax()
df['Y'].iloc[0:first_zero_idx] += 1


# In[17]:


#30. Consider a DataFrame containing rows and columns of purely numerical data. Create a list of the row-column index locations of the 3 largest values.

df = pd.DataFrame(np.random.RandomState(30).randint(1, 101, size=(8, 8)))
print(df)
df.unstack().sort_values()[-3:].index.tolist()


# In[18]:


#31. Given a DataFrame with a column of group IDs, 'grps', and a column of corresponding integer values, 'vals', replace any negative values in 'vals' with the group mean.

df = pd.DataFrame({"vals": np.random.RandomState(31).randint(-30, 30, size=15), 
                   "grps": np.random.RandomState(31).choice(["A", "B"], 15)})
print(df)
print("\n")
def replace(group):
    mask = group<0
    group[mask] = group[~mask].mean()
    return group

df.groupby(['grps'])['vals'].transform(replace)


# In[19]:


#32. Implement a rolling mean over groups with window size 3, which ignores NaN value. For example consider the following DataFrame:
df = pd.DataFrame({'group': list('aabbabbbabab'),
                       'value': [1, 2, 3, np.nan, 2, 3, 
                                 np.nan, 1, 7, 3, np.nan, 8]})
print(df)
print("\n")
g1 = df.groupby(['group'])['value']              # group values  
g2 = df.fillna(0).groupby(['group'])['value']    # fillna, then group values

s = g2.rolling(3, min_periods=1).sum() / g1.rolling(3, min_periods=1).count() # compute means

s.reset_index(level=0, drop=True).sort_index()  # drop/sort index


# In[20]:


#33. Create a DatetimeIndex that contains each business day of 2015 and use it to index a Series of random numbers. Let's call this Series s.

dti = pd.date_range(start='2015-01-01', end='2015-12-31', freq='B') 
s = pd.Series(np.random.rand(len(dti)), index=dti)
print(s)


# In[21]:


#34. Find the sum of the values in s for every Wednesday.
s[s.index.weekday == 2].sum()


# In[22]:


#35. For each calendar month in s, find the mean of values.

s.resample('M').mean()


# In[23]:


#36. For each group of four consecutive calendar months in s, find the date on which the highest value occurred.

s.groupby(pd.TimeGrouper('4M')).idxmax()


# In[24]:


#37. Create a DateTimeIndex consisting of the third Thursday in each month for the years 2015 and 2016.

pd.date_range('2015-01-01', '2016-12-31', freq='WOM-3THU')


# In[25]:


#38. Some values in the the FlightNumber column are missing. These numbers are meant to increase by 10 with each row so 10055 and 10075 need to be put in place. Fill in these missing numbers and make the column an integer column (instead of a float column).

df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', ' (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})
print(df)
print("\n")
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)
print(df)


# In[26]:


#39. The From_To column would be better as two separate columns! Split each string on the underscore delimiter _ to give a new temporary DataFrame with the correct values. Assign the correct column names to this temporary DataFrame.

temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']
temp


# In[27]:


#40. Notice how the capitalisation of the city names is all mixed up in this temporary DataFrame. Standardise the strings so that only the first letter is uppercase (e.g. "londON" should become "London".)

temp['From'] = temp['From'].str.capitalize()
temp['To'] = temp['To'].str.capitalize()
temp


# In[28]:


#41. Delete the From_To column from df and attach the temporary DataFrame from the previous questions.

df1 = df.drop(df.columns[[0]], axis=1) 
print(df1)


# In[29]:


df3 = pd.concat([temp, df1], axis=1, join='inner')
print(df3)


# In[30]:


#42. In the Airline column, you can see some extra puctuation and symbols have appeared around the airline names. Pull out just the airline name. E.g. '(British Airways. )' should become 'British Airways'.

df3['Airline'] = df3['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip()
print(df3)


# In[31]:


#43. In the RecentDelays column, the values have been entered into the DataFrame as a list. We would like each first value in its own column, each second value in its own column, and so on. If there isn't an Nth value, the value should be NaN.
#Expand the Series of lists into a DataFrame named delays, rename the columns delay_1, delay_2, etc. and replace the unwanted RecentDelays column in df with delays.

delays = df3['RecentDelays'].apply(pd.Series)
delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns)+1)]
df3 = df3.drop('RecentDelays', axis=1).join(delays)
print(df3) 


# In[32]:


#44. Given the lists letters = ['A', 'B', 'C'] and numbers = list(range(10)), construct a MultiIndex object from the product of the two lists. Use it to index a Series of random numbers. Call this Series s.

letters = ['A', 'B', 'C']
numbers = list(range(10))

mi = pd.MultiIndex.from_product([letters, numbers])
s = pd.Series(np.random.rand(30), index=mi)
s


# In[33]:


#45. Check the index of s is lexicographically sorted (this is a necessary proprty for indexing to work correctly with a MultiIndex).

s.index.lexsort_depth == s.index.nlevels


# In[34]:


#46. Select the labels 1, 3 and 6 from the second level of the MultiIndexed Series.

s.loc[:, [1, 3, 6]]


# In[35]:


#47. Slice the Series s; slice up to label 'B' for the first level and from label 5 onwards for the second level.

s.loc[pd.IndexSlice[:'B', 5:]]
#or
s.loc[slice(None, 'B'), slice(5, None)]


# In[36]:


#48. Sum the values in s for each label in the first level (you should have Series giving you a total for labels A, B and C).

s.sum(level=0)


# In[37]:


#49. Suppose that sum() (and other methods) did not accept a level keyword argument. How else could you perform the equivalent of s.sum(level=1)?
s.unstack().sum(axis=0)


# In[38]:


#50. Exchange the levels of the MultiIndex so we have an index of the form (letters, numbers). Is this new Series properly lexsorted? If not, sort it.

new_s = s.swaplevel(0, 1)

# check
new_s.index.is_lexsorted()

# sort
new_s = new_s.sort_index()
new_s


# In[48]:


#51
X = 5
Y = 4

p = pd.core.reshape.util.cartesian_product([np.arange(X), np.arange(Y)])
df = pd.DataFrame(np.asarray(p).T, columns=['x', 'y'])
df


# In[49]:


#52
df['mine'] = np.random.binomial(1, 0.4, X*Y)
df


# In[50]:


#53
df['adjacent'] =     df.merge(df + [ 1,  1, 0], on=['x', 'y'], how='left')      .merge(df + [ 1, -1, 0], on=['x', 'y'], how='left')      .merge(df + [-1,  1, 0], on=['x', 'y'], how='left')      .merge(df + [-1, -1, 0], on=['x', 'y'], how='left')      .merge(df + [ 1,  0, 0], on=['x', 'y'], how='left')      .merge(df + [-1,  0, 0], on=['x', 'y'], how='left')      .merge(df + [ 0,  1, 0], on=['x', 'y'], how='left')      .merge(df + [ 0, -1, 0], on=['x', 'y'], how='left')       .iloc[:, 3:]        .sum(axis=1)
        
# An alternative solution is to pivot the DataFrame 
# to form the "actual" grid of mines and use convolution.
# See https://github.com/jakevdp/matplotlib_pydata2013/blob/master/examples/minesweeper.py

from scipy.signal import convolve2d

mine_grid = df.pivot_table(columns='x', index='y', values='mine')
counts = convolve2d(mine_grid.astype(complex), np.ones((3, 3)), mode='same').real.astype(int)
df['adjacent'] = (counts - mine_grid).ravel('F')


# In[51]:


#54
df.loc[df['mine'] == 1, 'adjacent'] = np.nan


# In[52]:


#55
df.drop('mine', axis=1).set_index(['y', 'x']).unstack()


# In[40]:


#56
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})

df.plot.scatter("xs", "ys", color = "black", marker = "x")


# In[41]:


#57
df = pd.DataFrame({"productivity":[5,2,3,1,4,5,6,7,8,3,4,8,9],
                   "hours_in"    :[1,9,6,5,3,9,2,9,1,7,4,2,2],
                   "happiness"   :[2,1,3,2,3,1,2,3,1,2,2,1,3],
                   "caffienated" :[0,0,1,1,0,0,0,0,1,1,0,1,0]})

df.plot.scatter("hours_in", "productivity", s = df.happiness * 30, c = df.caffienated)


# In[42]:


#58
df = pd.DataFrame({"revenue":[57,68,63,71,72,90,80,62,59,51,47,52],
                   "advertising":[2.1,1.9,2.7,3.0,3.6,3.2,2.7,2.4,1.8,1.6,1.3,1.9],
                   "month":range(12)
                  })

ax = df.plot.bar("month", "revenue", color = "green")
df.plot.line("month", "advertising", secondary_y = True, ax = ax)
ax.set_xlim((-1,12))


# In[53]:


import numpy as np
def float_to_time(x):
    return str(int(x)) + ":" + str(int(x%1 * 60)).zfill(2) + ":" + str(int(x*60 % 1 * 60)).zfill(2)

def day_stock_data():
    #NYSE is open from 9:30 to 4:00
    time = 9.5
    price = 100
    results = [(float_to_time(time), price)]
    while time < 16:
        elapsed = np.random.exponential(.001)
        time += elapsed
        if time > 16:
            break
        price_diff = np.random.uniform(.999, 1.001)
        price *= price_diff
        results.append((float_to_time(time), price))
    
    
    df = pd.DataFrame(results, columns = ['time','price'])
    df.time = pd.to_datetime(df.time)
    return df

#Don't read me unless you get stuck!
def plot_candlestick(agg):
    """
    agg is a DataFrame which has a DatetimeIndex and five columns: ["open","high","low","close","color"]
    """
    fig, ax = plt.subplots()
    for time in agg.index:
        ax.plot([time.hour] * 2, agg.loc[time, ["high","low"]].values, color = "black")
        ax.plot([time.hour] * 2, agg.loc[time, ["open","close"]].values, color = agg.loc[time, "color"], linewidth = 10)

    ax.set_xlim((8,16))
    ax.set_ylabel("Price")
    ax.set_xlabel("Hour")
    ax.set_title("OHLC of Stock Value During Trading Day")
    plt.show()


# In[54]:


#59
df = day_stock_data()
df.head()


# In[56]:


df.set_index("time", inplace = True)
agg = df.resample("H").ohlc()
agg.columns = agg.columns.droplevel()
agg["color"] = (agg.close > agg.open).map({True:"green",False:"red"})
agg.head()


# In[57]:


#60
plot_candlestick(agg)


# In[ ]:




