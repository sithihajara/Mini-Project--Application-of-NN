# Mini-Project--Application-of-NN

## Project Title:
Stock market prediction

## Project Description 
We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%.
Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.

## Algorithm:
1. import the necessary pakages.
2. install the csv file
3. using the for loop and predict the output
4. plot the graph 
5. analyze the regression bar plot

## Google Colab Link:
https://colab.research.google.com/drive/1rknMNlbLphgS6ObhFSfGUWMfB-S_kPIE?usp=sharing
## Program:
import the necessary pakages
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
```
install the csv file
```
df = pd.read_csv('/content/Tesla.csv')
df.head()
```
```
df.shape
df.describe()
df.info()
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
df.head()
df[df['Close'] == df['Adj Close']].shape
df = df.drop(['Adj Close'], axis=1)
df.isnull().sum()
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.distplot(df[col])
plt.show()
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.boxplot(df[col])
plt.show()
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()
plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
features = df[['open-close', 'low-high']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
models = [LogisticRegression(), SVC(
kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

print(f'{models[i]} : ')
print('Training Accuracy : ', metrics.roc_auc_score(
	Y_train, models[i].predict_proba(X_train)[:,1]))
print('Validation Accuracy : ', metrics.roc_auc_score(
	Y_valid, models[i].predict_proba(X_valid)[:,1]))
print()
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()

```
## Output:
![205438173-9196c1d9-d1fa-4e74-b266-d23fee3401bd](https://user-images.githubusercontent.com/94219582/205443127-7cd91267-4d59-429a-92fc-60c1117871a9.png)

![205438196-596a76e9-86c8-47ea-8187-c13564263d45](https://user-images.githubusercontent.com/94219582/205443131-2e681411-0011-4386-b13a-441b5d80a33f.png)

![205438259-95f88ec2-b61e-4716-b1da-8b05c0d3b0e3](https://user-images.githubusercontent.com/94219582/205443149-358f6783-c8e3-4b1f-8f21-d95a9865c598.png)


![205438320-23d2e985-9b69-4df8-b772-75fbe0922d7c](https://user-images.githubusercontent.com/94219582/205443154-36de9d08-cfce-468b-8f52-d62d15fd2a03.png)


![205438384-d878e425-d958-4533-9cb7-c993fb6604ae](https://user-images.githubusercontent.com/94219582/205443207-7dbb5343-eeba-4a18-bb35-d5b46444d4d3.png)

![205438400-6ad9ce27-78a6-413c-a283-965f5d39890b](https://user-images.githubusercontent.com/94219582/205443208-0f44c489-b8fe-45ad-83cf-126ac2035b25.png)

![205438546-e1d94bc0-c469-4934-bb74-dac982f57320](https://user-images.githubusercontent.com/94219582/205443232-f5453114-229e-4ebc-adc9-f67892650159.png)

![205438568-19925be5-6105-4027-9474-f1d3942f4aaa](https://user-images.githubusercontent.com/94219582/205443245-cc6e6e02-a712-46be-b9f4-9f59217ac74f.png)


![205438690-1e1ae744-67a8-40c0-a1e4-6b897732cbbc](https://user-images.githubusercontent.com/94219582/205443284-82f95e08-1c58-420f-9887-a7ec6a3ac61a.png)

![205438720-4a2f6508-2079-43dd-be33-5190caeccf53](https://user-images.githubusercontent.com/94219582/205443299-812c9cd9-9263-4b11-b35f-d063099211b0.png)

![205438746-790653fa-ec49-4be9-bb90-fccba9a60cc0](https://user-images.githubusercontent.com/94219582/205443307-810331eb-07c1-40fa-8474-796a638cd4b0.png)

![205438768-4365e122-bef2-42d8-9afc-73ee968b9d94](https://user-images.githubusercontent.com/94219582/205443315-8fcb40e5-dfb7-4952-8f75-a986ca2b132a.png)

![205438796-b2e22e65-0ebe-4a05-8524-4b38f86c26e9](https://user-images.githubusercontent.com/94219582/205443320-c0b2e010-3746-4ebf-a845-bbde34f94407.png)

## Advantage :
Python is the most popular programming language in finance. 
Because it is an object-oriented and open-source language, it is used by many large corporations,
including Google, for a variety of projects. Python can be used to import financial data such as
stock quotes using the Pandas framework.

## Result:
Thus, stock market prediction is implemented successfully.
