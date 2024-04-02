#Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



#Downloads 
nltk.download('punkt')
nltk.download('stopwords')
df = pd.read_csv('Downloads/COMP 429/Project/text.csv')



#EDA
df.head()

#Shape of data
df.shape #(416809, 3)

#Get label counts
df.label.value_counts()

#Check for null values
df.isnull().sum() #NA

#Restructre data
df.rename(columns={'text': 'Text', 'label': 'Label'}, inplace=True)
# Dropping the Index Colums
df.drop('Unnamed: 0',axis=1,inplace=True)
#0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'
df['Label'] = df['Label'].replace(0,'Sadness')
df['Label'] = df['Label'].replace(1,'Joy')
df['Label'] = df['Label'].replace(2,'Love')
df['Label'] = df['Label'].replace(3,'Anger')
df['Label'] = df['Label'].replace(4,'Fear')
df['Label'] = df['Label'].replace(5,'Surprise')
df.head()



#Data visualization

#Create a bar plot
count = df['Label'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=count.index, y=count.values, palette="viridis")
plt.title('Count of Each Label')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()



#Data Cleaning
#Remove URLs
df['Text'] = df['Text'].str.replace(r'http\S+', '', regex=True)

#Remove extra whitespaces
df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex=True)

#Remove numeric values
df['Text'] = df['Text'].str.replace(r'\d+', '', regex=True)

#Remove special characters and punctuation
df['Text'] = df['Text'].str.replace(r'[^\w\s]', '', regex=True)

#Lowercasing
df['Text'] = df['Text'].str.lower()

#Remove stop words
stop = stopwords.words('english')
df["Text"] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Check
print(df.head())


#Training/Development/Test Split

















