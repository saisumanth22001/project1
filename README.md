1st   wap to use matplotlib and plot the graph

import pandas as pd

df = pd.read_excel("C:/Users/pdwiv/OneDrive/Desktop/excel-comp-data.xlsx")
df.head()

df["total"] = df["Jan"] + df["Feb"] + df["Mar"]
df.head()
import matplotlib.pyplot as plt
df['total'].plot(kind="hist")
plt.xlabel("X axis label")
plt.ylabel("Y axis label")
plt.title("Histogram Plot")
plt.show()
df['total'].plot()
plt.show()
category_data = df["account"]
total_data = df["total"]
plt.pie(total_data, labels=category_data, autopct='%1.1f%%')
plt.title("% of total sales of each account")
plt.show()

2nd
Sentiment analysis using nlp

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')


texts = ["I love working with Python!", 
         "The weather is terrible today.",
         "This movie is absolutely amazing!",
         "I don't like this restaurant at all."]


analyzer = SentimentIntensityAnalyzer()


for text in texts:
    sentiment = analyzer.polarity_scores(text)
    print("Text: ", text)
    print("Sentiment Score: ", sentiment['compound'])
    print("\n")


3rd              
twitter  analysis using vadar library
import pandas as pd
from textblob import TextBlob


data = pd.read_csv('sample_data.csv')

for index, row in data.iterrows():
    text = row['text']
    sentiment = TextBlob(text).sentiment.polarity
    print("Text: ", text)
    print("Sentiment Score: ", sentiment)
    print("\n")

4 working on opencv for image processing

import cv2

img = cv2.imread('image.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale Image', gray_img)
cv2.waitKey(0)

threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]

cv2.imshow('Thresholded Image', threshold_img)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, 
cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.imshow('Image with Contours', img)
cv2.waitKey(0)

cv2.imwrite('image_with_contours.jpg', img)


print a 

for i in range(10):

  for j in range(5):

    if ((i==0) ):

      if ((j==0) or(j==4)):

        print(" ",end=" ")

      else:

        print("*",end=" ")

    elif  (i==4):

      print("*",end=" ")

    else:

      if ((j==0) or (j==4)) :

        print("*",end=" ")

      else:

        print(" ",end=" ")

  print("\n")



