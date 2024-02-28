#!/usr/bin/env python
# coding: utf-8

# # Step-Wise Regression Project

# By: Ashton passmore

# In[1]:


# installing lahman package 
# import sys
# !{sys.executable} -m pip install tq-lahman-datasets


# In[2]:


# importing required packages
from teqniqly.lahman_datasets import LahmanDatasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import statsmodels.api as sm


# ## Background and Problem Definition

# For project 2 I will be doing the same thing as project 1 which is using step wise linear regression to try and predict how many homeruns a team will give up in a given year. This time around I also want to use step wise linear regression to try and answer the question of how many games will a team win in a given year. I will be using the same package as before which is the lahman package. I will be using the teams dataset from that package. The lahman teams data set has 2985 observations and 48 variables and it gives yearly statistics for Major League Baseball teams from 1871 - 2021. Also this time around I'm going to expand my year range because last time the sample size was a bit small and made the model not as precise as it could of been with more information. I'm going to use the year range 1994-2022 since this is the start of the steriod era and goes to the present day.I also want to try and use the wins model to see if it can accuratly predict the amount of wins a team currently has in 2023.

# ## Data Wrangling, Munging and Cleaning

# In[3]:


# making the data frame 
ld = LahmanDatasets()
df_names = ld.dataframe_names
ld.load()
teams_df = ld["Teams"]


# In[4]:


# making sure the dataframe loaded correctly
teams_df.head()


# In[5]:


# filering the data frame to only give the year range 1994 - 2022.
revised_teams = teams_df[teams_df['yearID'] >= 1994]
revised_teams.head()


# ## Exploratory Data Analysis

# In[6]:


# plotting the revised data frame as a histogram and scatter plot of home runs allowed to see the distributin of the data
fig = px.scatter(revised_teams, x = 'yearID', y = 'HRA', color = 'franchID')
fig.update_layout(title = "Scatter plot of Home Runs Agianst by Year",
                 xaxis_title = "Year",
                 yaxis_title = "Home Runs Agianst")
fig.show()

fig2 = px.histogram(revised_teams, x = 'HRA')
fig2.update_layout(title = "Histogram of Home Runs Against",
                  xaxis_title = "Home Runs Agianst")
fig2.update_traces(marker_line_width = 1, marker_line_color = "deeppink")
fig2.show()


# In[7]:


# plotting the revised data frame as a histogram and scatter plot of wins to see the distributin of the data
fig = px.scatter(revised_teams, x = 'yearID', y = 'W', color = 'franchID')
fig.update_layout(title = "Scatter plot of Wins by Year",
                 xaxis_title = "Year",
                 yaxis_title = "Wins")
fig.show()

fig2 = px.histogram(revised_teams, x = 'W')
fig2.update_layout(title = "Histogram of Wins",
                  xaxis_title = "Wins")
fig2.update_traces(marker_line_width = 1, marker_line_color = "deeppink")
fig2.show()


# After plotting the data frame I forgot about the 2020 season which was only 60 games and doesn't provide an accurate sample size for the year so i'm going to remove it from the data frame.

# In[8]:


# making a new revised data frame with 2020 excluded
revised_teams2 = revised_teams[revised_teams['yearID'] != 2020]
revised_teams2.head()


# In[9]:


# plotting the revised data frame W/O 2020 as a histogram and scatter plot of home runs allowed to see the distributin of the data
fig = px.scatter(revised_teams2, x = 'yearID', y = 'HRA', color = 'franchID')
fig.update_layout(title = "Scatter plot of Home Runs Agianst by Year",
                 xaxis_title = "Year",
                 yaxis_title = "Home Runs Agianst")
fig.show()

fig2 = px.histogram(revised_teams2, x = 'HRA')
fig2.update_layout(title = "Histogram of Home Runs Against",
                  xaxis_title = "Home Runs Agianst")
fig2.update_traces(marker_line_width = 1, marker_line_color = "deeppink")
fig2.show()


# In[10]:


# plotting the revised data frame W/O 2020 as a histogram and scatter plot of the teams ranks to see the distributin of the data
fig = px.scatter(revised_teams2, x = 'yearID', y = 'W', color = 'franchID')
fig.update_layout(title = "Scatter plot of Wins by Year",
                 xaxis_title = "Year",
                 yaxis_title = "Wins")
fig.show()

fig2 = px.histogram(revised_teams2, x = 'W')
fig2.update_layout(title = "Histogram of Wins",
                  xaxis_title = "Wins")
fig2.update_traces(marker_line_width = 1, marker_line_color = "deeppink")
fig2.show()


# Both wins and home runs aginast look to be normally distributed with home runs agianst lookin a little left skewed.

# ### Building the Model

# I'm using the same process as project 1 to build my linear regression model. I'm splitting the data up into two different sets one set that is 80% of the data for training the model and the other 20% for testing the model at the end.

# #### setting up the test and training set

# In[11]:


np.random.seed(1234)
# training set with 80% of total data
train = revised_teams2.sample(frac=0.8)
# test set with remaining 20% of the data
test = revised_teams2.drop(train.index)
# checking to make sure everything seperated properly
print(revised_teams2.shape[0])
print(train.shape[0])
print(test.shape[0])
# the number of rows in the train and test sets add up to the rows in our main data set so we are all good


# In order to answer the questions from the beggining I'm going to need to set up two models. One will be for predicting Home Runs Agianst (HRA) like my orginal project and the other will answer the additional question of predicting how many wins (W) a team will have in a given season. Both models with use step wise linear reggression to make the predictions.

# ### Home Runs Agianst Model

# ###### Choosing Independent Variables

# For the first model our dependant variable will be Home Runs Agianst (HRA). When looking at the data set any stats that deal with pitching have some sort of relevance to home runs against since you can only give up home runs when your team is on defense. For my independent variables I’m choosing pretty much all of the pitching variables because they could all have an effect on home runs against. I’m choosing Wins(W), Losses(L), Runs Against(RA), Earned Runs (ER), Earned Run Average (ERA), Complete Games(CG), Shut Outs (SHO), Saves(SV), Outs Pitched (IPouts), Hits against (HA), Walks Against (BBA), and finally Strike Outs Against (SOA).

# In[12]:


indVars = ['W','L','RA','ER','ERA','CG','SHO','SV','IPouts','HA','BBA','SOA']
depVar = 'HRA'
HRAfit = sm.OLS(train[depVar], train[indVars]).fit()
HRAfit.summary()


# now I'll perform step-wise regression to improve the model (get all variables 0.05 p values and lower)

# In[13]:


# taking L out of indVars because it is the least signifigant variable then I will remake the fit.
indVars.remove("L")
HRAfit2 = sm.OLS(train[depVar], train[indVars]).fit()
HRAfit2.summary()


# In[14]:


# taking SV out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("SV")
HRAfit3 = sm.OLS(train[depVar], train[indVars]).fit()
HRAfit3.summary()


# In[15]:


# taking CG out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("CG")
HRAfit4 = sm.OLS(train[depVar], train[indVars]).fit()
HRAfit4.summary()


# In[16]:


# taking W out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("W")
HRAfit5 = sm.OLS(train[depVar], train[indVars]).fit()
HRAfit5.summary()


# In[17]:


# taking RA out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("RA")
HRAfit6 = sm.OLS(train[depVar], train[indVars]).fit()
HRAfit6.summary()


# In[18]:


# taking ERA out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("ERA")
HRAfit7 = sm.OLS(train[depVar], train[indVars]).fit()
HRAfit7.summary()


# ###### Interpretation
# Now that all independent variables are at or below 0.05 they are all signifigant and fit 7 is our final fit.
# 
# The R-squared value tells us how well independent variables fit our dependent variables the closer to 1 the
# better and the closer to 0 is bad. The value of 0.991 is good and tells us that about 99% of our
# outputs can be explained and about 1% can’t be.
# 

# In[19]:


res = HRAfit7.resid


# In[20]:


fig = px.box(res)
fig.update_layout(title = "Boxplot of Residuals",
                  yaxis_title = "Residual Values")
fig.show()


# In[21]:


plt.scatter(HRAfit7.fittedvalues, res, color = "deeppink")
plt.plot([min(HRAfit7.fittedvalues), max(HRAfit7.fittedvalues)], [0,0])
plt.xlabel('Home Runs Agianst (HRA)')
plt.ylabel('Residual')
plt.show()


# In[22]:


fig = px.histogram(res)
fig.update_layout(title = "Histogram of Residuals",
                 xaxis_title = "Residuals",
                 yaxis_title = "Frequency")
fig.update_traces(marker_line_width = 1, marker_line_color = "white")
fig.show()


# The boxplot shows us that our model is a good fit for our data because the median is close to 0 and Q1 and Q3 seem to be about the same length. This is further backed up by the histogram because our residuals seem to be normally distributed.

# ### Wins Model

# ###### Choosing Independent Variables

# The process for the second model will be pretty similar to the the process for the first model. I'll the second model I'll choose some independant variables that I think have an effect on the amount a wins a team has and then remove variables that have a p-value greater than 0.05. For the second model our dependant variable will be Wins (W). For the independent variables I'm choosing pretty much every hitting, pitching, and fielding variable because they all could have an impact on a teams a win total. For the independent variables I'm chosing: losses(L), runs(R), hits(H), doubles(2B), triples(3B), homeruns(HR), walks(BB), strikeouts(SO), stolen bases(SB), caught stealing(CS), sacrifice flys(SF), runs agianst(RA), earned runs(ER), earned run average(ERA), complete games(CG), shut outs(SHO), saves(SV), outs pitched(IPouts), hits aginast(HA), home runs agianst(HRA), walks agianst (BBA), strike outs agianst(SOA), errors(E), double plays(DP), and fielding percentage(FP).

# In[23]:


indVars = ['L','R','H','2B','3B','HR','BB','SO','SB','CS','SF','RA','ER','ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP','FP']
depVar = 'W'
Wfit = sm.OLS(train[depVar], train[indVars]).fit()
Wfit.summary()


# In[24]:


# taking SO out of indVars because it is the least signifigant variable then I will remake the fit.
indVars.remove("SO")
Wfit2 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit2.summary()


# In[25]:


# taking CG out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("CG")
Wfit3 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit3.summary()


# In[26]:


# taking HRA out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("HRA")
Wfit4 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit4.summary()


# In[27]:


# taking ER out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("ER")
Wfit5 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit5.summary()


# In[28]:


# taking ERA out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("ERA")
Wfit6 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit6.summary()


# In[29]:


# taking 3B out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("3B")
Wfit7 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit7.summary()


# In[30]:


# taking CS out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("CS")
Wfit8 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit8.summary()


# In[31]:


# taking SF out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("SF")
Wfit9 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit9.summary()


# In[32]:


# taking 2B out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("2B")
Wfit10 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit10.summary()


# In[33]:


# taking DP out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("DP")
Wfit11 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit11.summary()


# In[34]:


# taking SOA out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("SOA")
Wfit12 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit12.summary()


# In[35]:


# taking HR out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("HR")
Wfit13 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit13.summary()


# In[36]:


# taking SB out of indVars because it is the next least signifigant variable then I will remake the fit.
indVars.remove("SB")
Wfit14 = sm.OLS(train[depVar], train[indVars]).fit()
Wfit14.summary()


# ###### interpretation

# Now that all the independent variables are at or below 0.05 they are all signifigant and fit14 is the final fit.
# The R-squared value tells us how well our independent variables fit our dependent variables the closer to 1 the better and the closer to 0 is bad. The value of 1 is as good as it gets and tells us that all of our outputs can be explained.

# In[37]:


res = Wfit14.resid


# In[38]:


fig = px.box(res)
fig.update_layout(title = "Boxplot of Residuals",
                 yaxis_title = "Residual Values")
fig.show()


# In[39]:


plt.scatter(Wfit14.fittedvalues, res, color = "green")
plt.plot([min(Wfit14.fittedvalues), max(Wfit14.fittedvalues)],[0,0])
plt.xlabel("Wins (W)")
plt.ylabel("Residual")
plt.show()


# In[40]:


fig = px.histogram(res)
fig.update_layout(title = "Histogram of Residuals",
                 xaxis_title = "Residuals",
                 yaxis_title = "Frequency")
fig.update_traces(marker_line_width = 1, marker_line_color = "orange")
fig.show()


# The boxplot shows us that our model is a good fit for our data because the median is close to 0 and Q1 and Q3 seem to be about the same length. This is further backed up by the histogram because our residuals seem to be normally distributed.

# ## Data Visualization

# In[41]:


# applying both predictive models with the test set
HRAindVars = ['ER','SHO','IPouts','HA','BBA','SOA']
WindVars = ['L','R','H','BB','RA','SHO','SV','IPouts','HA','BBA','E','FP']
predictionsHRA = HRAfit7.predict(test[HRAindVars])
predictionsW = Wfit14.predict(test[WindVars])
print(predictionsHRA.head())
print(predictionsW.head())


# In[42]:


warnings.filterwarnings("ignore")
predictDF = test.copy()
predictDF['HRApredictions'] = predictionsHRA
predictDF['Wpredictions'] = predictionsW
RevisedTeams2DF = predictDF[['yearID', 'franchID','W', 'HRA', 'HRApredictions','Wpredictions']]
RevisedTeams2DF['roundedHRAPredictions'] = RevisedTeams2DF['HRApredictions'].round(0)
RevisedTeams2DF['roundedWPredictions'] = RevisedTeams2DF['Wpredictions'].round(0)
RevisedTeams2DF['HRATF'] = RevisedTeams2DF['HRA'] == RevisedTeams2DF['roundedHRAPredictions']
RevisedTeams2DF['WTF'] = RevisedTeams2DF['W'] == RevisedTeams2DF['roundedWPredictions']
RevisedTeams2DF.head()
# I rounded the predictions up because you can't have half a win or half a home


# #### Graphs of Predicted Values vs Actual Values Using Test Set

# ###### HRA preditions grpah

# In[43]:


fig = px.scatter(RevisedTeams2DF, x = 'roundedHRAPredictions', y = 'HRA', color = 'franchID')
fig.update_layout(title = "Home Runs Agianst Values V. Predicted Home Runs Aginast Values",
                 xaxis_title = "Predicted Home Runs Agianst",
                 yaxis_title = "Home Runs Agianst")
fig.add_shape(type = "line",
             line=dict(color="red", width=2),
             x0 = 100,
             y0=100,
             x1=275,
             y1=275)
fig.show()


# ###### Wins predictions graph

# In[44]:


fig = px.scatter(RevisedTeams2DF, x = 'roundedWPredictions', y = 'W', color = 'franchID')
fig.update_layout(title = "Wins Values V. Predicted Wins Values",
                 xaxis_title = "Predicted Wins",
                 yaxis_title = "Wins")
fig.add_shape(type = "line",
             line=dict(color="red", width=2),
             x0 = 40,
             y0=40,
             x1=125,
             y1=125)
fig.show()


# The Home Runs Against model seems to give a rough estimate of how many home runs a team may give up while to Wins model seems to accurately predict a teams actual win total.

# ###### Results

# In[45]:


HRAResults = RevisedTeams2DF.groupby("HRATF").size().reset_index(name="count")
print(HRAResults)
WResults = RevisedTeams2DF.groupby("WTF").size().reset_index(name="count")
print(WResults)


# The Home Runs Agianst Model predicted 11/166 values correctly which is a little over 6% and isn't all that great while the Wins model predicted 54/166 values correctly which is around 33% and is pretty good especially when looking at the graph and seeing the predcited values are typically within 5 wins of the actual value.

# ## Predicting 2023 wins so far

# In order to get the 2023 stats that are avaliable so far I ended up downloading the csv files of team data from baseball reference and imported them into excel to clean rather than doing web scarapping and cleaning in python.

# In[46]:


# importing the csv file
teamStats2023 = pd.read_csv("2023TeamStats.csv")
teamStats2023.head()


# In[47]:


WindVars = ['L','R','H','BB','RA','SHO','SV','IPouts','HA','BBA','E','FP']
Wpredictions2023 = Wfit14.predict(teamStats2023[WindVars])
print(Wpredictions2023.head())


# In[48]:


predictDF = teamStats2023.copy()
predictDF['Wpredictions'] = Wpredictions2023
RevisedTeams2DF = predictDF[['franchID','W','Wpredictions']]
RevisedTeams2DF['roundedWPredictions'] = RevisedTeams2DF['Wpredictions'].round(0)
RevisedTeams2DF['TF'] = RevisedTeams2DF['W'] == RevisedTeams2DF['roundedWPredictions']
RevisedTeams2DF.head()


# In[49]:


fig = px.scatter(RevisedTeams2DF, x = 'roundedWPredictions', y = 'W', color = 'franchID')
fig.update_layout(title = "Wins Values V. Predicted Wins Values in 2023",
                 xaxis_title = "Predicted Wins",
                 yaxis_title = "Wins")
fig.add_shape(type = "line",
             line=dict(color="red", width=2),
             x0 = 0,
             y0=0,
             x1=40,
             y1=40)
fig.show()


# In[50]:


WResults = RevisedTeams2DF.groupby("TF").size().reset_index(name="count")
print(WResults)


# The model didn't predict the right amount of wins that a team currently has but the predictions were really close and this is probably due to the low sample size since the 2023 season just started.

# # Conclusion

# In conclusion, I was able to answer all of the questions I wanted to. Even though the models I made weren't as good as I were expecting both models give you a good idea of how many wins a team may have or home runs agianst a team may give up. Overall I really don't think I could improve on either model with the information I used in the packages but, if I had more time I think I could've  pulled more advanced baseball statistics and been able to get more accuracy out of both models. In the end both models did what they were supposed to and the predicted values give a good indication of what the real value will be.

# # References

# https://www.baseball-reference.com/leagues/majors/2023.shtml#all_teams_standard_pitching
