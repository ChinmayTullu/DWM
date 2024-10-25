import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\tanis\\Downloads\\DWM\\spotify_songs.csv")
df.describe()
df['track_name'] = df['track_name'].fillna('Unknown Track')
df['track_artist'] = df['track_artist'].fillna('Unknown Artist')
df['track_album_name'] = df['track_album_name'].fillna('Unknown Album')


# Removing outliers using IQR
columns_for_IQR=['track_popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
for i in columns_for_IQR:
    Q1=df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    IQR=Q3-Q1
    lowerbound = Q1-1.5*IQR
    upperbound = Q3+1.5*IQR
    df = df[(df[i]>lowerbound) & (df[i]<upperbound)]
    
    
#Data Discretization - Quantile Binning
from sklearn.preprocessing import KBinsDiscretizer

kbin_value = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
columns_for_binning=['track_popularity', 'danceability', 'energy', 'key', 'loudness',
'speechiness','acousticness', 'liveness', 'valence', 'tempo','duration_ms']

for i in columns_for_binning:
    df[i]=kbin_value.fit_transform(df[[i]])
    
    
# Normalization using Min-Max
columns_for_minmax=['track_popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1) )
for i in columns_for_minmax:
    df[[i]] = scaler.fit_transform(df[[i]])
    
    
# Data Visualization
numeric_columns=['track_popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
for col in numeric_columns:
    plt.figure(figsize=(4, 3))
    plt.boxplot(df[col])
    plt.xlabel(col, fontsize=8)
    plt.ylabel('Value', fontsize=8)
    plt.title(f'Boxplot of {col}', fontsize=10)
    plt.xticks(fontsize=8) # Adjust tick labels
    plt.yticks(fontsize=8) # Adjust tick labels
    plt.tight_layout() # Prevent clipping
    plt.show()



df_numeric=df[['track_popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']]
features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
for feature in features:
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x=df[feature], y=df['track_popularity'])
    plt.title(f'{feature} vs Track Popularity')
    plt.xlabel(feature)
    plt.ylabel('Track Popularity')
    plt.show()



corr = df_numeric.corr()
corr_matrix = df_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlOrBr', annot_kws={"size": 10}, cbar=True)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


for col in df_numeric:
    plt.figure(figsize=(4, 3))
    plt.hist(df_numeric[col], edgecolor='black')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col}')
    plt.show()