import pandas as pd
from datetime import datetime
def load_data(path):
    """
    Loads the data from path, and returns it as a pandas dataframe
    :param path: this is the csv file path
    :return: Pandas dataframe
    """
    df = pd.read_csv(path)
    return df

def add_new_columns(df):
    """
    Adds new columns to the dataframe
    :param df: this is the dataframe
    :return: Pandas dataframe with added columns
    """
    season_name = {
        0: 'spring',
        1: 'summer',
        2: 'fall',
        3: 'winter',
    }
    df['season_name'] = df['season'].apply(lambda x: season_name[x])
    df['Hour'] = df['timestamp'].apply(lambda ts: datetime.strptime(ts, "%d/%m/%Y %H:%M").hour)
    df['Day'] = df['timestamp'].apply(lambda ts: datetime.strptime(ts, "%d/%m/%Y %H:%M").day)
    df['Month'] = df['timestamp'].apply(lambda ts: datetime.strptime(ts, "%d/%m/%Y %H:%M").month)
    df['Year'] = df['timestamp'].apply(lambda ts: datetime.strptime(ts, "%d/%m/%Y %H:%M").year)
    df['is_weekend_holiday'] = df.apply(lambda row: 2 * row['is_holiday'] + row['is_weekend'] + 1, axis=1)
    df['t_diff'] = df.apply(lambda row: row['t2'] - row['t1'], axis=1)
    return df

def data_analysis(df):
    """
    Analyses the data and prints the result
    :param df: this is the dataframe
    """
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()
    corr = df.corr()
    pairs = []
    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            c2 = cols[j]
            val = corr.at[c1, c2]
            abs_val = abs(val)
            pairs.append(((c1, c2), abs_val))

    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    highest5 = sorted_pairs[:5]
    non_zero_pairs = [p for p in sorted_pairs if p[1] > 0]
    lowest5 = sorted(non_zero_pairs, key=lambda x: x[1])[:5]

    print("Highest correlated are: ")
    for i, ((f1, f2), abs_val) in enumerate(highest5, 1):
        print(f"{i}. ('{f1}', '{f2}') with {abs_val:.6f}")

    print("\nLowest correlated are: ")
    for i, ((f1, f2), abs_val) in enumerate(lowest5, 1):
        print(f"{i}. ('{f1}', '{f2}') with {abs_val:.6f}")

    season_avg_dict = df.groupby('season_name')['t_diff'].mean().round(2).to_dict()
    all_avg = round(df['t_diff'].mean(), 2)

    print()
    print(f"fall average t_diff is {season_avg_dict['fall']}")
    print(f"spring average t_diff is {season_avg_dict['spring']}")
    print(f"summer average t_diff is {season_avg_dict['summer']}")
    print(f"winter average t_diff is {season_avg_dict['winter']}")
    print(f"All average t_diff is {all_avg :.2f}")