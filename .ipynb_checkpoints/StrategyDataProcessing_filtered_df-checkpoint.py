import pandas as pd 

df = pd.read_csv("stock_data.csv")

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)
date_from = pd.to_datetime('1970-01-01', format='%Y-%d-%m')
number_corp = df.loc[date_from:].dropna(axis=1).shape[1]
number_date = df.loc[date_from:].dropna(axis=1).shape[0]
# Searchf for the best 

date_founded = True 
if not(date_founded):
    for i in range(0,len(df) - 2000):

        date_from += pd.Timedelta(days=1)

        number_corp_new = df.loc[date_from:].dropna(axis=1).shape[1]
        number_date = df.loc[date_from:].dropna(axis=1).shape[0]

        if number_corp_new > number_corp:
            print(date_from, number_corp, (number_corp*number_date))
            number_corp = number_corp_new

# From the analysis, we draw the conclusion to select 2000-01-01 to exploit the max number of stocks and data
final_date = pd.to_datetime('2000-01-01', format='%Y-%d-%m')
df = df.loc[final_date:].dropna(axis=1)

df.to_csv("data_cleaned.csv")