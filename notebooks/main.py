#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load the dataset
df= pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\AIR_QUALITY\dataset\air_quality.csv")
df.head()
#basic inspection
df.info()
df.describe()
df.isnull().sum()
df.columns

# Replace NA with NaN and convert to numeric
df[['pollutant_min', 'pollutant_max', 'pollutant_avg']] = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']].replace('NA', pd.NA)
for col in ['pollutant_min', 'pollutant_max', 'pollutant_avg']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#drop rows with all  three pollutant values as NaN
df = df.dropna(subset=['pollutant_min', 'pollutant_max', 'pollutant_avg'], how='all')

#Check duplicates
duplicates= df.duplicated(subset=['station','pollutant_id','last_update'])
print(f"Number of duplicates: {duplicates.sum()}")

#Outlier check using z_score
pm25_avrage =df[df['pollutant_id']=='PM2.5']['pollutant_avg']
mean, std = pm25_avrage.mean() , pm25_avrage.std()
outliers= pm25_avrage[(pm25_avrage > mean + 3*std) | (pm25_avrage <mean - 3*std)]
print(f"Potential PM2.5 outliers: {outliers}")

#checking for invalid geographic data
#geospatial validation
validlatitute =df['latitude'].between(8,37)  #India's latitude range
validlong= df['longitude'].between(68,97) # India's longitude range
print(f"Invalid coordinates: {((~validlatitute) | (~validlong)).sum()}")

#save the cleaned data to a new csv file
df.to_csv('../dataset/air_quality_cleaned.csv', index=False)
print("Cleaned data is  saved!")

df = pd.read_csv('../dataset/air_quality_cleaned.csv')
print("Data loaded successfully!")
df.head()


# Objective 1: Analyze pollutant distribution across cities/states.***
#Summarize  pollutant distribution
pollutant_summary = df.groupby('pollutant_id')[['pollutant_avg']].agg(['mean', 'min','max','count'])
print("Pollutant summary statistics\n")
pollutant_summary

plt.figure(figsize=(12, 6))
sns.boxplot(x='pollutant_id', y='pollutant_avg',hue='pollutant_id', data=df, palette='Set2', legend=False)
plt.title('Distribution of all Pollutants')
plt.xlabel('Pollutants')
plt.ylabel('Average Concentration (µg/m³)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Top and bottom PM2.5 cities
pm25_avg = df[df['pollutant_id'] == 'PM2.5'].groupby('city')['pollutant_avg'].mean().sort_values()
print("Lowest PM2.5 cities:\n", pm25_avg.head(5))
print("Highest PM2.5 cities:\n", pm25_avg.tail(5))

#Objective 2: Identifying worst air quality regions(Top 10 most polluted cities).***
#Top 10 PM2.5 cities
pm25_avg = df[df['pollutant_id'] == 'PM2.5'].groupby('city')['pollutant_avg'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
pm25_avg.head(10).sort_values().plot(kind='barh', color='salmon')
plt.title('Top 10 Most Polluted Cities by PM2.5 ')
plt.xlabel('Average PM2.5 (µg/m³)')
plt.ylabel('City')
plt.tight_layout()
plt.show()

#Objective 3: Compare air quality between northern and southern regions of India.***
#North vs. South comparison
north = ['Delhi', 'Uttar_Pradesh', 'Bihar', 'Haryana']
south = ['Karnataka', 'TamilNadu', 'Kerala', 'Andhra_Pradesh']
df['region']= df['state'].apply(lambda x: 'North' if x in north else ('South' if x in south else 'Other'))
pm25_region = df[df['pollutant_id']== 'PM2.5'].groupby('region')['pollutant_avg'].mean()
print("PM2.5 by region:\n", pm25_region)


#regional PM2.5 comparison
plt.figure(figsize=(8,5))
pm25_region.plot(kind='bar', color=['blue','green', 'gray'])
plt.title('Average PM2.5 by Region')
plt.xlabel('Region')
plt.ylabel('Average PM2.5 (µg/m³)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#PM2.5 variability across stations
plt.figure(figsize=(8,6))
sns.boxplot(x='pollutant_id', y='pollutant_avg',data= df[df['pollutant_id']=='PM2.5'])
plt.xlabel('Pollutant')
plt.ylabel('Average PM2.5 (µg/m³)')
plt.tight_layout()
plt.show()

#Advanced Analysis and Mapping***
import folium
north =['Delhi', 'Uttar_Pradesh', 'Bihar', 'Haryana']
south= ['Karnataka', 'TamilNadu', 'Kerala', 'Andhra_Pradesh']
df['region'] = df['state'].apply(lambda x: 'North' if x in north else ('South' if x in south else 'Other'))
pm25_region_stats = df[df['pollutant_id'] == 'PM2.5'].groupby('region')['pollutant_avg'].agg(['mean', 'std', 'count'])
print("PM2.5 regional stats:\n", pm25_region_stats)
#regional comparison with error bars
plt.figure(figsize=(8,6))
sns.barplot(x='region', y='pollutant_avg', hue='region',data= df[df['pollutant_id']=='PM2.5'], errorbar='sd' , palette =['blue','green', 'gray'], legend = False)
plt.title('Average PM2.5 by Region with Variability')
plt.xlabel('Region')
plt.ylabel('Average PM2.5 (µg/m³)')
plt.tight_layout()
plt.show()


# Objective 4: Investigate correlations between different pollutants.***
#Pollutant correlations
pivotdf = df.pivot_table(index=['state', 'city', 'station'], columns='pollutant_id', values='pollutant_avg', aggfunc='mean')
corr = pivotdf.corr()
print("Correlation Matrix:\n")
corr
#Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Between Pollutants ')
plt.tight_layout()
plt.savefig('../outputs/corr_heatmap.png')
plt.show()

#Objective 4: Visualize the geographical distribution of pollution hotspots.***
#Mapping PM2.5 hotspots
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  
pm25_data = df[df['pollutant_id'] == 'PM2.5'].dropna(subset=['latitude', 'longitude'])
for _, row in pm25_data.iterrows():
    color = 'red' if row['pollutant_avg'] > 100 else 'yellow' if row['pollutant_avg'] > 50 else 'green'
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"{row['city']}, {row['state']}: {row['pollutant_avg']} µg/m³"
    ).add_to(m)
m.save('../outputs/air_quality_map.html')
print("Map saved as 'air_quality_map.html' in outputs  folder")


#Objective 5: Propose a simplified Air Quality Index (AQI) score for each city based on combined pollutant metrics.***
from sklearn.preprocessing import MinMaxScaler
#Simplified AQI score
key_pollutants = ['PM2.5', 'PM10', 'NO2']
aqi_df = df[df['pollutant_id'].isin(key_pollutants)].pivot_table(index='city', columns='pollutant_id', values='pollutant_avg', aggfunc='mean')
#simple fill method "My approach i chose means over interpolation for simplicity"
aqi_df = aqi_df.fillna(aqi_df.mean())  
scaler = MinMaxScaler()
aqi_scaled = scaler.fit_transform(aqi_df)
#Feature engineering
aqi_df['AQI_Score'] = aqi_scaled.mean(axis=1) * 100  
print("Top 5 Cities by AQI Score:\n", aqi_df.sort_values('AQI_Score', ascending=False).head())


#Top 10 AQI cities
plt.figure(figsize=(10, 6))
aqi_df['AQI_Score'].sort_values(ascending=False).head(10).plot(kind='bar', color=plt.cm.viridis(np.linspace(0, 1, 10)))
plt.title('Top 10 Cities by Simplified AQI Score')
plt.xlabel('City')
plt.ylabel('AQI Score (0-100)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Infer location types from station names
df['location_type'] = df['station'].apply(
    lambda x: 'Industrial' if 'Zone' in str(x) or 'Plant' in str(x) 
    else 'Urban' if 'Nagar' in str(x) or 'Colony' in str(x) 
    else 'Other'
)
pm25_by_type = df[df['pollutant_id'] == 'PM2.5'].groupby('location_type')['pollutant_avg'].agg(['mean', 'count'])
print("PM2.5 by location type:\n", pm25_by_type)

#PM2.5 by location type

pm25_by_type = df[df['pollutant_id'] == 'PM2.5'].groupby('location_type')['pollutant_avg'].mean()
plt.figure(figsize=(8, 6))
plt.pie(pm25_by_type, labels=pm25_by_type.index, colors=['orange', 'teal', 'gray'], autopct='%1.1f%%')
plt.title(' PM2.5 Share by Location Type')
plt.tight_layout()

plt.show()