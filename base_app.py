"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
from PIL import Image
import string

# Model dependencies
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Vectorizer
news_vectorizer = open("resources/tfidfvectorize.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Change Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Introduction","Information", "Exploratory Data Analysis","Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)

	#Building out the "Introduction" page
	if selection == "Introduction":
		st.info("Introduction")
		intro_image = Image.open('resources/imgs/climate_change.jpg')
		st.image(intro_image, use_column_width=True)
		st.markdown("""Climate change is the phenomenon of an increasing number of greenhouse gases within the earth's atmosphere that is accompanied by major shifts in weather patterns. 
						This is largely human-induced and is as a result of increased levels of atmospheric carbon dioxide produced by the use of fossil fuels for basic living neccesities as well as large industrial processes. 
						The effects of climate change affect the livelihoods of both people and animals and is experienced via intense drought, storms, heat waves, rising sea levels, melting glaciers and warming oceans, 
						Furthermore, as climate change worsens, dangerous weather events are becoming more frequent or severe.""")

		st.markdown("""Over several years, many companies have attempted to implemt startegies around lessening their environmental impact or carbon footprint. 
					They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. 
					However, problems they experience include guaging how their products may be recieved based on people's views and opinions of climate change. 
					By determining how people perceive climate change and whether or not they believe it is a real threat, these companies can improve on their market research efforts. 
					Additionally, access to a large collection of consumer sentiments that also spans multiple demographic and geographic categories will influence company insights 
					and their future marketing strategies allowing them to find the right target market to direct their products and efforts toward.""")

		st.markdown("""In the context of climate change and sustainable companies, Team_TS1 aim to provide a means for such companies to determine the views or sentiments of people towards climate change. 
					To do this, tweets pertaining to climate change will be looked at and used to train a classification model in order to accurately classify the opinions behind those tweets, into those who believe in climate change and those who do not. 
					This notebook details the work flow of Team_TS1 in building, training and assessing different classifier models to provide a suitable solution that can be implemented in future marketing strategies of climate concious companies.""")

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("""The purpose of this predict is to build a classification model that can accurately place a Tweeter account in one of four class based on their Tweets about Climate Change. 
					These classes are Anti-Climate change which is denoted by -1, Neutral about Climate Change which is denoted by 0, Pro-Climate Change which is denoted by 1 and News reporting about Climate change denoted by 2.""")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']].head()) # will write the df to the page

    #Building out the "Exploratory Data Analysis" page
	if selection == "Exploratory Data Analysis":
		st.info("Visualizing Data in Pre-processing")
		st.markdown("**Distribution of the Response**")
		st.markdown("""The distribution of the label variable (sentiment) provides insight into the frequencies per category of the sentiment (-1 to 2) expressed by each tweet.
		 			This gives an indication of the more popular vs unpopular opinions on climate change. 
		 			Given the categorical nature of this variable a countplot provides the best insight into the frequencies for each category.""")
		senti_image = Image.open('resources/imgs/senti_distribution.png')
		st.image(senti_image, use_column_width=True)

		st.markdown("""The distribution of sentiments shows a clear difference in the frequencies observed for each sentiment, with the 'Pro' climate change tweets making up majority of the opinions expressed within this data set at over 8000 counts. 
					'Anti' climate changes tweets make up the lowest number of opinons with just over 1000 counts and with more people having a neutral response towards climate change than dibelieving in it. 
					The number of News/factual tweets is higher than both that of 'Anti' and 'Neutral' views over 3000 views.""")

		st.markdown("The following table of value counts further describes the difference in frequency between each sentiment.")
		dist_image = Image.open('resources/imgs/number_of_distributions_table.png')
		st.image(dist_image, use_column_width=False)

		st.markdown("**Table of the Tokenized data**")
		st.image(Image.open('resources/imgs/tokens.png'),caption=None, use_column_width=True)

		st.markdown("**Table of the lemmatized data**")
		st.image(Image.open('resources/imgs/lemma.png'),caption=None, use_column_width=True)

		st.markdown("**WordCloud of most used words in each Sentiment**")
		st.markdown("Exploring which words are used most frequently in each sentiment provides valuable information about what topics hold significant weight and importance in each class")

		if st.checkbox("Most Frequently used words"):
			sentiment = st.radio("Choose a sentiment",("anti-climate change","pro-climate change", "neutral on climate change","news reports on climate change"))
			if sentiment == 'pro-climate change':
				st.image(Image.open('resources/imgs/word_pro.png'),caption=None, use_column_width=True)
				st.markdown("""Even though the word Climate Change is the most used word in all for classes, the interesting part are the words used together with it in tweets. These words in this particular sentiment include believe climate, fight climate, change real.
							All these words suggest that people who are pro Climate Change strongly believe it is a problem that can be addressed and slowed down.""")

			if sentiment == 'anti-climate change':
				st.image(Image.open('resources/imgs/word_anti.png'),caption=None, use_column_width=True)
				st.markdown("""Some of frequently used words in tweets about Climate Change from individuals who believe human activity has no effect of Climate Change include: Climate Change, Global Warming, Liberal, Science, Hoax.
							The bigger words are more used than the smaller ones. It makes sense that a wordcloud about Climate Change whether pro or anti will have Climate Change as the most used word.""")

			if sentiment == 'neutral on climate change':
				st.image(Image.open('resources/imgs/word_neutral.png'),caption=None, use_column_width=True)
				st.markdown("""The frequently used words in this sentiment suggest that people who are neutral about Climate Change tend to tweet about both anti and pro Climate Change.""")

			if sentiment == 'news reports on climate change':
				st.image(Image.open('resources/imgs/word_news.png'),caption=None, use_column_width=True)
				st.markdown("""The words use frequently by news outlets do not differ from those used by people who believe Climate Change is influenced by human activity and thus can be slowed down by change in behaviour e.g manifucturing less plastic""")
		
		st.markdown("**Extracting hashtags**")	
		st.markdown("""Hashtags are used in tweets when one wants to refer to or bring to light a particular topic. When a user clicks on a hashtag on Twitter all the tweets related to that particular topic appear. 
					They are worth considering for our sentiment classes as we will be able to observe which phrases/hashtags are used most by each of our four groups. """)
			

		if st.checkbox('Hashtags'):
			sentiment = st.radio("Choose a sentiment",("anti-climate change","pro-climate change", "neutral on climate change","news reports on climate change"))
			if sentiment == 'pro-climate change':	
				hashtag_image = Image.open('resources/imgs/pro_hashtags.png')
				st.image(hashtag_image, use_column_width=True)
				st.markdown(""" The frequetly used hashtags by individuals who are pro Climate Change include: climatechange. BeforeTheFlood, ImVotingBecause, COP22 and Paris Aggrement. It is to be expected for pro Climate Change individuals to tweet about this topic,
							Before the Flood is a docuentary highlighting enviromental degradation that leads to Global Warning and Climate Change. This documentary raised concern for people who believe in human influence on Climate Change as the USA had pulled out of the Paris Agreement. 
							They were seeing the possible approaching catastrophe while being aware that the USA will not be an active participant in delaying this catastrophe. This in turn motivated more people to go and vote in the USA presedential Elections to bring in an administration that would take a more active approach in tackling Climate Change.""")

			if sentiment == 'news reports on climate change':	
				st.image(Image.open('resources/imgs/news_hashtags.png'), caption=None, use_column_width=True)
				st.markdown("""News reports use the following hashtags most frequently: climate, environment, climate change, Trump and news. This is to be expected as the main topic is Climate Change,
							news outlets will report factual information related to the topic, consulating with experts in the field. They would also report on anti Climate Change stance when it is strongly held by a president of a country such as USA, hence the Trump hashtag. """)

			if sentiment == 'neutral on climate change':
				st.image(Image.open('resources/imgs/neutral_hashtags.png'), caption=None, use_column_width=True)
				st.markdown("""Individuals who take a neutral stance on Climate Change tend to use the following hashtags: climatechange, Trump, BeforeTheFlood and ParisAccod. 
							This seems to suggest that people who are neautral consume information from both extremities about the Climate Change topic, information from Climate Change believers and Non believers. """)

			if sentiment == 'anti-climate change':
				st.image(Image.open('resources/imgs/anti_hashtags.png'), caption=None, use_column_width=True)
				st.markdown("""Some of the most frequently used hashtags from individuals who do not believe that climate change is influenced by human activity include: MAGA, Trump, fakenews and ClimateScam. 
							MAGA was a campaign slogan for the then USA presidential candidate Donald Trump, it stands for Make America Greate Again. This would suggest that some of his supporters do not believe in climate change.
							This is not surprising as former president Trump pulled the USA out of the Paris Climate Accord. It also ties in with the belief that climate change is a scam and it is fake news.""")

		st.markdown("")			

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","researchers say that climate change will be going on for three years")

		

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/rfc.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

    	
		
	
	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
