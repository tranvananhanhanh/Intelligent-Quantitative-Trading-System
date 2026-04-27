import pandas as pd
import os

for ticker in ['SPY', 'QQQ']:
    filepath = f'data/fmp_daily/{ticker}_daily.csv'
    sep = '=' * 60
    print(f'{chr(10)}{sep}')
    print(f'{ticker}_daily.csv')
    print(f'{sep}')
    print(f'File exists: {os.path.exists(filepath)}')
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f'Shape: {df.shape}')
        print(f'Columns: {list(df.columns)}')
        
        # Find date column
        date_col = None
        for col in ['datadate', 'Date', 'date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            print(f'Date range: {min_date} to {max_date}')
            
            print(f'{chr(10)}Last 5 rows:')
            last_rows = df[[date_col, 'close']].tail(5)
            for idx, row in last_rows.iterrows():
                close_val = row['close']
                print(f'  {row[date_col]} - Close: {close_val}')
        else:
            print('No date column found')
    else:
        print('File not found')
