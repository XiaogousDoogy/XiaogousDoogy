#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 21:23:40 2022

@author: Group 5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from math import pi
from sklearn.preprocessing import MinMaxScaler
import pyarrow
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from kmodes.kprototypes import KPrototypes
---------------------------------------------------------------------------------------------------------------------------------------------------------
''' DataSet Input'''

df = pd.read_stata("android_xsection_treated_cleaned (1).dta")
---------------------------------------------------------------------------------------------------------------------------------------------------------
''' Data Explore Analysis'''

df['treatment'] = df.treatment.astype("category")

df_1 = pd.DataFrame({
    'conversion_mean': df.groupby('treatment', as_index=True).mean().conversion,
    'revenue_mean': df.groupby('treatment', as_index=True).mean().revenue,
    'purchases_mean': df.groupby('treatment', as_index=True).mean().purchases,
    'sessions_mean': df.groupby('treatment', as_index=True).mean().sessions,
    'rounds_played': df.groupby('treatment', as_index=True).mean().rounds_played
}
)

df_1 = df_1.reindex(['after0days', 'after25days', 'after50days', 'no_promo'])

df_1.reset_index(inplace=True)

df_1[['rounds_played', 'sessions_mean']] = np.log(df_1[['rounds_played', 'sessions_mean']])
df_1['conversion_mean'] = np.multiply(df_1['conversion_mean'], 100)
df_1['purchases_mean'] = np.multiply(df_1['purchases_mean'], 10)

categories = list(df_1)[1:]
N = len(categories)

values = df_1.loc[0].drop('treatment').values.flatten().tolist()
values += values[:1]

df['treatment'] = df.treatment.astype("category")
df_agg = pd.DataFrame({
    'conversion_mean': df.groupby('treatment', as_index=True).mean().conversion,
    'conversion_se': df.groupby('treatment', as_index=True).sem().conversion,
    'revenue_mean': df.groupby('treatment', as_index=True).mean().revenue,
    'revenue_se': df.groupby('treatment', as_index=True).sem().revenue,
    'purchases_mean': df.groupby('treatment', as_index=True).mean().purchases,
    'purchases_se': df.groupby('treatment', as_index=True).sem().purchases,
    'sessions_mean': df.groupby('treatment', as_index=True).mean().sessions,
    'sessions_se': df.groupby('treatment', as_index=True).sem().sessions,
    'rounds_mean': df.groupby('treatment', as_index=True).mean().rounds_played,
    'rounds_se': df.groupby('treatment', as_index=True).sem().rounds_played,
}
)
df_agg = df_agg.reindex(
    ['no_promo', 'after0days', 'after25days', 'after50days'])
df_agg.reset_index(inplace=True)

# Plot for conversion
plt.figure(figsize=(8, 5))
plt.bar(df_agg.treatment, df_agg.conversion_mean, color='blue',
        width=0.8)
plt.errorbar(df_agg.treatment, df_agg.conversion_mean,
             yerr=1.96 * df_agg.conversion_se,
             fmt='_',
             color='black',
             ecolor='black',
             elinewidth=2,
             capsize=4
             )
plt.ylabel("conversion_mean", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot for revenue
plt.figure(figsize=(8, 5))
plt.bar(df_agg.treatment, df_agg.revenue_mean, color='red',
        width=0.8)
plt.errorbar(df_agg.treatment, df_agg.revenue_mean,
             yerr=1.96 * df_agg.revenue_se,
             fmt='_',
             color='black',
             ecolor='black',
             elinewidth=2,
             capsize=4
             )
plt.ylabel("revenue_mean", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot for purchases
plt.figure(figsize=(8, 5))
plt.bar(df_agg.treatment, df_agg.purchases_mean, color='green',
        width=0.8)
plt.errorbar(df_agg.treatment, df_agg.purchases_mean,
             yerr=1.96 * df_agg.purchases_se,
             fmt='_',
             color='black',
             ecolor='black',
             elinewidth=2,
             capsize=4
             )
plt.ylabel("purchases_mean", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

X = df[['t0', 't25', 't50']]
Y = df['purchases']
X_with_c = sm.add_constant(X)
Reg = sm.OLS(Y, X_with_c)
Reg_ = Reg.fit()
print(Reg_.summary())

# plot for sessions
plt.figure(figsize=(8, 5))
plt.bar(df_agg.treatment, df_agg.sessions_mean, color='yellow',
        width=0.8)
plt.errorbar(df_agg.treatment, df_agg.sessions_mean,
             yerr=1.96 * df_agg.sessions_se,
             fmt='_',
             color='black',
             ecolor='black',
             elinewidth=2,
             capsize=4
             )
plt.ylim(140, 200)
plt.ylabel("sessions_mean", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

#plot for rounds_played
plt.figure(figsize=(8, 5))
plt.bar(df_agg.treatment, df_agg.rounds_mean, color='purple',
        width=0.8)
plt.errorbar(df_agg.treatment, df_agg.rounds_mean,
             yerr=1.96 * df_agg.rounds_se,
             fmt='_',
             color='black',
             ecolor='black',
             elinewidth=2,
             capsize=4
             )
plt.ylabel("rounds_mean", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Set the data for Rader Chart
df['treatment'] = df.treatment.astype("category")
df_1 = pd.DataFrame({
    'conversion_mean': df.groupby('treatment',as_index=True).mean().conversion,
    'revenue_mean': df.groupby('treatment',as_index=True).mean().revenue,
    'purchases_mean': df.groupby('treatment',as_index=True).mean().purchases,
    'active_hours': df.groupby('treatment',as_index=True).mean().active_hours,
    'rounds_played': df.groupby('treatment',as_index=True).mean().rounds_played
    }
    )


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# dataset after log function 

df_test = df_1

df_test.loc['zero'] = [0, 0, 0, 0, 0]

df_log2 = NormalizeData(df_test)

df_log2.reset_index(inplace = True)


# Radar Chart
'''After 0 days'''
# number of variable
categories=list(df_log2)[1:]
N = len(categories)
 
# repeat the first value to close the circular graph
values=df_log2.loc[0].drop('treatment').values.flatten().tolist()
values += values[:1]
values


# the angle of each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
# ax.set_rlabel_position(0)
plt.yticks([0.5, 0.75,  1], ["0.5","0.75", "1"], color="grey", size=7)
plt.ylim(0,1)
ax.set_yticklabels([])
plt.title("After0days")
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

# Show the graph
plt.show()



'''after 25 days'''
# number of variable
categories=list(df_log2)[1:]
N = len(categories)
 
# repeat the first value to close the circular graph
values = df_log2.loc[1].drop('treatment').values.flatten().tolist()
values += values[:1]
values
 
# the angle of each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
# ax.set_rlabel_position(0)
plt.yticks([0.5, 0.75, 1], ["0.5","0.75", "1"], color="grey", size=7)
plt.ylim(0,1)
ax.set_yticklabels([])
plt.title("After25days")
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

# Show the graph
plt.show()


'''after 50 days'''
# number of variable
categories=list(df_log2)[1:]
N = len(categories)
 
# repeat the first value to close the circular graph
values = df_log2.loc[2].drop('treatment').values.flatten().tolist()
values += values[:1]
values
 
# the angle of each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
# ax.set_rlabel_position(0)
plt.yticks([0.5, 0.75, 1], ["0.5","0.75", "1"], color="grey", size=7)
plt.ylim(0,1)
ax.set_yticklabels([])
plt.title("After50days")
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

# Show the graph
plt.show()


'''no promo'''
# number of variable
categories=list(df_log2)[1:]
N = len(categories)
 
# repeat the first value to close the circular graph
values = df_log2.loc[3].drop('treatment').values.flatten().tolist()
values += values[:1]
values


# the angle of each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
# ax.set_rlabel_position(0)
plt.yticks([0.5, 0.75, 1], ["0.5","0.75", "1"], color="grey", size=7)
plt.ylim(0,1)
ax.set_yticklabels([])
plt.title("No Promotion")
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)

# Show the graph
plt.show()
---------------------------------------------------------------------------------------------------------------------------------------------------------
'''Trade off Testing'''

#linear regression analysis for active hours and purchases under promotion
regression7 = smf.ols(formula='purchases~ c*active_hours', data=df) 
model7 = regression7.fit()
model7.summary()
# The code above shows that there is no treat off. Thus we tried to prove the trade off in the other way in the following codes.

#'c': 'Indicator for no promotions treatment (not until day 180 of lifetime)',
#Created a metric 't':'Indicator for promotions treatment (not until day 180 of lifetime)
t = pd.DataFrame(np.where(df['c']==0,1,0),columns = ['t'])
df = pd.concat([df,t], axis=1, join="inner")
#linear regression analysis for the active hours and purchases under promotion
regression7 = smf.ols(formula='purchases~ t+ active_hours+ t*active_hours', data=df)
model7 = regression7.fit()
model7.summary()
# the result shows that the p value of 't' is far greater than 0.05, which proves this metric is not significant.
# In conclusion, there is no sufficient statistical results to prove trade off between active hours and purchase under promotion

---------------------------------------------------------------------------------------------------------------------------------------------------------
''' Phase 1 : whats the revenue perofrmance with promotios'''

X = df[['t0', 't25', 't50']]
Y = df['revenue']
X_with_c = sm.add_constant(X)
Reg = sm.OLS(Y, X_with_c)
Reg_ = Reg.fit()
print(Reg_.summary())

'''The pvalue tells us only the feedback of t25 and t0 group represents the population attribute. we could explain this
 as the revenue change only occured at population level when we apply promotion after 0 days(P= 0.08) and 25 days
 (P = 0.01), implying that early promotion seems a better option rather than late.'''

# Is revenue good enough?
# The revenue from promotion
X = df[['t0', 't25', 't50']]
Y_p_rev = df['promo_revenue']
X_with_c = sm.add_constant(X)
Reg_p_rev = sm.OLS(Y_p_rev, X_with_c)
Reg_p_rev_ = Reg_p_rev.fit()
print(Reg_p_rev_.summary())
# The revenue from non-promotion (statistically not significant)
X = df[['t0', 't25', 't50']]
Y_np_rev = df['shop_revenue']
X_with_c = sm.add_constant(X)
Reg_np_rev = sm.OLS(Y_np_rev, X_with_c)
Reg_np_rev_ = Reg_np_rev.fit()
print(Reg_np_rev_.summary())
'''Statistically, logically and rationally, revenue from promotion is a sure thing, and eailer a promotion started,
better the revenue generated'''

---------------------------------------------------------------------------------------------------------------------------------------------------------
''' Phase 2 : what makes people consume? impulse buying? how to measure this one? '''
# bottom logic of promotion, stimuate users to consume, also known as, impulse buying from users' perspective
''' ch1_complete: analyzing users
    revenue_ch1: willingless of spending money on the game without any promotion
    active_hours_ch1: degree of preference of such a game
    Impulse_lvl_N: revenue_ch1 / active_hours_ch1;'''


df['days_till_fist_promo_purchase'] = df.days_first_promo_purchase - df.days_first_promo_offer

#df['Impulsiveness'] = df.log_promo_revenue/(df.promo_purchases+1) * np.log((df.days_first_purchase /(df.days_till_fist_promo_purchase
                                                                                                                              + 1))+1)
# Robustness Check
df['Impulsiveness'] = ''
df['Impulsiveness'] =(df.log_revenue/df.purchases) * (1/(df.days_till_fist_promo_purchase+1))


df.Impulsiveness.replace(np.nan, 0, inplace=True)
df.Impulsiveness.mean()

---------------------------------------------------------------------------------------------------------------------------------------------------------
'''Insert Implusiveness into the DataSet, and prepare for Segmentation'''
# Lets consider impulse_lvl_N as one indictator
df['treatment'] = df.treatment.astype("category")
df_agg_new = pd.DataFrame({
    'conversion_mean': df.groupby('treatment', as_index=True).mean().conversion,
    'conversion_se': df.groupby('treatment', as_index=True).sem().conversion,
    'revenue_mean': df.groupby('treatment', as_index=True).mean().revenue,
    'revenue_se': df.groupby('treatment', as_index=True).sem().revenue,
    'purchases_mean': df.groupby('treatment', as_index=True).mean().purchases,
    'purchases_se': df.groupby('treatment', as_index=True).sem().purchases,
    'sessions_mean': df.groupby('treatment', as_index=True).mean().sessions,
    'sessions_se': df.groupby('treatment', as_index=True).sem().sessions,
    'Impulsiveness_mean': df.groupby('treatment', as_index=True).mean().Impulsiveness,
    'Impulsiveness_se': df.groupby('treatment', as_index=True).sem().Impulsiveness
}
)
df_agg_new = df_agg_new.reindex(
    ['no_promo', 'after0days', 'after25days', 'after50days'])
df_agg_new.reset_index(inplace=True)
plt.figure(figsize=(8, 5))
plt.bar(df_agg_new.treatment, df_agg_new.Impulsiveness_mean, color='pink',
        width=0.8)
plt.errorbar(df_agg_new.treatment, df_agg_new.Impulsiveness_mean,
             yerr=1.96 * df_agg_new.Impulsiveness_se,
             fmt='_',
             color='black',
             ecolor='black',
             elinewidth=2,
             capsize=4
             )
plt.ylabel("Impulsive_level_mean", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# df.to_csv('df_6214_new.csv', index=False) #Data Export

X = df[['t0', 't25', 't50']]
Y = df['Impulsiveness']
X_with_c = sm.add_constant(X)
Reg = sm.OLS(Y, X_with_c)
Reg_ = Reg.fit()
print(Reg_.summary())

#Get the new dataframe for segmentation
#Before Robustness Check
df['Impulsiveness_T1'] = ''
df['Impulsiveness_T1'] = np.where(df['treatment'] == 'no_promo', 0,
                                 df['Impulsiveness_T1'])
df['Impulsiveness_T1'] = np.where(df['treatment'] == 'after0days', df.log_promo_revenue/(df.promo_purchases+1),
                                 df['Impulsiveness_T1'])
df['Impulsiveness_T1'] = np.where(df['treatment'] == 'after25days', df.log_promo_revenue/(df.promo_purchases+1),
                                 df['Impulsiveness_T1'])
df['Impulsiveness_T1'] = np.where(df['treatment'] == 'after50days', df.log_promo_revenue/(df.promo_purchases+1),
                                 df['Impulsiveness_T1'])
df.Impulsiveness_T1.replace(np.nan, 0, inplace=True)
df.Impulsiveness_T1.mean()
df = df.astype({"Impulsiveness_T1": 'float64'})
#After Robustness Check
df['Impulsiveness_T1'] = ''
df['Impulsiveness_T1'] = (df.log_revenue/df.purchases)
df.Impulsiveness_T1.replace(np.nan, 0, inplace=True)
df.Impulsiveness_T1.mean()
df = df.astype({"Impulsiveness_T1": 'float64'})

#Before Robustness Check
df['Impulsiveness_T2'] = ''
df['Impulsiveness_T2'] = np.where(df['treatment'] == 'no_promo', 0,
                                 df['Impulsiveness_T2'])
df['Impulsiveness_T2'] = np.where(df['treatment'] == 'after0days', np.log((df.days_first_purchase /(df.days_till_fist_promo_purchase
                                                                                                                              + 1))+1),
                                 df['Impulsiveness_T2'])
df['Impulsiveness_T2'] = np.where(df['treatment'] == 'after25days', np.log((df.days_first_purchase /(df.days_till_fist_promo_purchase
                                                                                                                              + 1))+1),
                                 df['Impulsiveness_T2'])
df['Impulsiveness_T2'] = np.where(df['treatment'] == 'after50days', np.log((df.days_first_purchase /(df.days_till_fist_promo_purchase
                                                                                                                              + 1))+1),
                                 df['Impulsiveness_T2'])
df.Impulsiveness_T2.replace(np.nan, 0, inplace=True)
df.Impulsiveness_T2.mean()
#After Robustness Check
df['Impulsiveness_T2'] = ''
df['Impulsiveness_T2'] = (1/(df.days_till_fist_promo_purchase+1))
df.Impulsiveness_T2.replace(np.nan, 0, inplace=True)
df.Impulsiveness_T2.mean()


#FARM TO WIN MATRICS
df['Delta_Hours_per_Round'] = ''
df['Delta_Hours_per_Round'] = np.where(df['treatment'] == 'no_promo', df.active_hours/(df.rounds_played+1) / (df.active_hours_ch1 /(df.rounds_ch1
                                                                                                                              +1)+1),
                                 df['Delta_Hours_per_Round'])
df['Delta_Hours_per_Round'] = np.where(df['treatment'] == 'after0days', df.active_hours/(df.rounds_played+1) /  (df.active_hours_ch1 /(df.rounds_ch1
                                                                                                                              +1)+1),
                                 df['Delta_Hours_per_Round'])
df['Delta_Hours_per_Round'] = np.where(df['treatment'] == 'after25days', df.active_hours/(df.rounds_played+1) /  (df.active_hours_ch1 /(df.rounds_ch1
                                                                                                                              +1)+1),
                                 df['Delta_Hours_per_Round'])
df['Delta_Hours_per_Round'] = np.where(df['treatment'] == 'after50days', df.active_hours/(df.rounds_played+1) /  (df.active_hours_ch1 /(df.rounds_ch1
                                                                                                                              +1)+1),
                                 df['Delta_Hours_per_Round'])
df.Delta_Hours_per_Round.replace(np.nan, 0, inplace=True)
df.Delta_Hours_per_Round.mean()
df = df.astype({"Delta_Hours_per_Round": 'float64'})

df.Delta_Hours_per_Round[df['treatment'] == 'no_promo'].mean()
df.Delta_Hours_per_Round[df['treatment'] == 'after0days'].mean()
df.Delta_Hours_per_Round[df['treatment'] == 'after25days'].mean()
df.Delta_Hours_per_Round[df['treatment'] == 'after50days'].mean()
---------------------------------------------------------------------------------------------------------------------------------------------------------
'''Segmentation (KPrototype)'''
# Data Sampling 
df_sample = df.sample(10000,random_state = 1)
df_sample['dpi'] = df_sample['dpi'].astype('object')
df_Kprototype = df_sample[['active_hours_ch1', 'revenue_ch1', 'cash_spent_ch1','dpi', 'mobile_source']]
mark_array = df_Kprototype.values
categorical_features_idx = list(range(3,5))

# Find out the best number of clusters. It usually took a long time to run the following code.
for cluster in range(2, 15):
    kprototype = KPrototypes(n_clusters=cluster, verbose=2, max_iter=20,random_state=1)
    kprototype.fit_predict(mark_array, categorical = categorical_features_idx)
    cost.append(kprototype.cost_)
plt.plot(range(2,15),cost)
plt.xlabel('No of clusters')
plt.ylabel('Cost')

# From the above code, we found out 4 clusters will be the best number of clusters to explain.
kprototype4 = KPrototypes(n_clusters=4, verbose=2, max_iter=20,random_state=1)
kprototype4.fit_predict(mark_array, categorical = categorical_features_idx)
df_Kprototype['Impulsiveness'] = df_sample['Impulsiveness'].values
df_Kprototype['Impulsiveness_T1'] = df_sample['Impulsiveness_T1'].values
df_Kprototype['Impulsiveness_T2'] = df_sample['Impulsiveness_T2'].values
df_Kprototype['Delta_Hours_per_Round'] = df_sample['Delta_Hours_per_Round'].values
df_sample['Label'] = kprototype4.labels_.astype('int')
df_Kprototype['Label'] = kprototype4.labels_.astype('int')
df_Kprototype.dtypes
df_Kprototype = df_Kprototype.reset_index(drop=True)

S = pd.DataFrame(df_Kprototype.describe())
S_Object = pd.DataFrame(df_Kprototype.describe(include='object'))
#cluster 0
C1 = pd.DataFrame(df_Kprototype.loc[df_Kprototype['Label']==0].describe())
C1_Object = pd.DataFrame(df_Kprototype.loc[df_Kprototype['Label']==0].describe(include='object'))
#cluster 1
C2 = pd.DataFrame(df_Kprototype.loc[df_Kprototype['Label']==1].describe())
C2_Object = pd.DataFrame(df_Kprototype.loc[df_Kprototype['Label']==1].describe(include='object'))
#cluster 2
C3 = pd.DataFrame(df_Kprototype.loc[df_Kprototype['Label']==2].describe())
C3_Object = pd.DataFrame(df_Kprototype.loc[df_Kprototype['Label']==2].describe(include='object'))
#cluster 3
C4 = pd.DataFrame(df_Kprototype.loc[df_Kprototype['Label']==3].describe())
C4_Object = pd.DataFrame(df_Kprototype.loc[df_Kprototype['Label']==3].describe(include='object'))

len(df.loc[(df['treatment'] == 'no_promo') & 
                  (df['Impulsiveness'] >= df['Impulsiveness'].mean())]) / 54883
len(df.loc[(df['treatment'] == 'after0days') & 
                  (df['Impulsiveness'] >= df['Impulsiveness'].mean())]) / 52625
len(df.loc[(df['treatment'] == 'after25days') & 
                  (df['Impulsiveness'] >= df['Impulsiveness'].mean())]) / 26526
len(df.loc[(df['treatment'] == 'after50days') & 
                  (df['Impulsiveness'] >= df['Impulsiveness'].mean())])
---------------------------------------------------------------------------------------------------------------------------------------------------------
# Segments Summary
'''
Sample
	active_hours_ch1	revenue_ch1   
mean	7.3993	0.0865987092256546
	Impulsiveness	Impulsiveness_T1	Impulsiveness_T2
mean	0.02867   	0.02381	            0.0416
	Delta_Hours_per_Round
mean	0.0117

Type 1 User: "Belivers"
They play a lot, and they spend a lot. Often comes with a better device. 
The best way to gather users like them are through paid channel.
They are extrenly impulsive under promotion, espically the average revenue they generate
per purchase.

Type 2 User: "Observsers"
They play slightly more than the simple average, and have spent more than the average.
The effect of promotion on users like them shall be still very effective.
Paid channel
On the other way of describling them, they are so called potenital customers

Type 3 User: "Apathies"
They play the least, they spend no money on the game.
Their impulsiveness towards promotion are the lowest.
Paid channel
Those users are often described as inactivite users.

Type 4 User: "neutral"
The biggest difference this group has from other groups, is users are more likely from
organic channel, meaning they got in touch with this game by themselves.
user performance is very close to type 3 users, inactivite. Whereas they amount of time
they spent before chapter 1 is very close to the sample average.

Different segments, different promotion strategies
A straightforwared price descrimation could cuase business ethic issue, but to wordplay
it out, making something originially bad to something good is essencial in business circumsance
Price descrimation is inappropriate, but a secret shop is not.

Secret Shop for Type 1 Users:
    Underlying logic: 
        Limited time interval: 48 hours counting down
        Slogan: "Exclusive sale, designed only for you!"
        Emphasis the Highest Face value products.
        Cash weighted more.
    exp:
    25% off on 14.99$ package 
    35% off on 29.99$ package 
    50% off on 59.99$ package
    60% off on 99.99$ package
    
Secret Shop for Type 2 Users:
    Underlying logic: 
        Limited time interval: 72 hours counting down
        Slogan: "Having a bit trouble? your special develiver just arrived!"
        Emphasis the intermidate Face value products.
        Coins, cash and energy evenly distrubited
    exp:
    20% off on 9.99$ package 
    30% off on 14.99$ package 
    40% off on 59.99$ package
    50% off on 29.99$ package
  
Secret Shop for Type 3 Users:
    Underlying logic: 
        Limited time interval: 168 hours counting down
        Slogan: "The only package you need!"
        Emphasis the lower Face value products, the purpose is to boost conversion
        Coins, cash and energy evenly distrubited
    exp:
    20% off on 29.99$ package 
    30% off on 14.99$ package 
    40% off on 9.99$ package
    50% off on 4.99$ package
    60% off on 2.99$ package

Secret Shop for Type 4 Users:
    Underlying logic: 
        Limited time interval: 120 hours counting down
        Slogan: "The only package you need!"
        Emphasis the lower Face value products, the purpose is to boost conversion
        Coins, cash and energy evenly distrubited
    exp:
    20% off on 59.99$ package 
    30% off on 29.99$ package 
    40% off on 14.99$ package
    50% off on 9.99$ package
    60% off on 4.99$ package


Now we have all the secret shop designed, when to apply them to users?
T0? T25? or T50?
The idea is to test out the amount of users who have impulsiveness more than the average.
T0 is the best.
Even though people might argue T0 is not generating the most revenue, whereas a user is more impulsive when
seeing promotion, more possibile we could stimuate them with promotion
'''


---------------------------------------------------------------------------------------------------------------------------------------------------------
'''Farm to win'''

'''
At where we already know users in treatment groups are playing less, we had our assumption:
    the in-game currency they spent helped them to get through all rounds they played but 
'''

df['Delta_CashSpend_per_Round'] = ''
df['Delta_CashSpend_per_Round'] = df.cash_spent/(df.rounds_played+1) / (df.cash_spent_ch1/(df.rounds_ch1+1)+1)
df.Delta_CashSpend_per_Round.replace(np.nan, 0, inplace=True)
df.Delta_CashSpend_per_Round.mean()

df.loc[df['treatment'] == 'no_promo'].Delta_CashSpend_per_Round.mean()
df.loc[df['treatment'] == 'after0days'].Delta_CashSpend_per_Round.mean()
df.loc[df['treatment'] == 'after25days'].Delta_CashSpend_per_Round.mean()
df.loc[df['treatment'] == 'after50days'].Delta_CashSpend_per_Round.mean()

df['Delta_CoinsSpend_per_Round'] = ''
df['Delta_CoinsSpend_per_Round'] = df.coins_spent/(df.rounds_played+1) /(df.coins_spent_ch1/(df.rounds_ch1+1)+1)
df.Delta_CoinsSpend_per_Round.replace(np.nan, 0, inplace=True)
df.Delta_CoinsSpend_per_Round.mean()

df.loc[df['treatment'] == 'no_promo'].Delta_CoinsSpend_per_Round.mean()
df.loc[df['treatment'] == 'after0days'].Delta_CoinsSpend_per_Round.mean()
df.loc[df['treatment'] == 'after25days'].Delta_CoinsSpend_per_Round.mean()
df.loc[df['treatment'] == 'after50days'].Delta_CoinsSpend_per_Round.mean()

df.Delta_CoinsSpend_per_Round.replace(np.nan, 0, inplace=True)
df.Delta_CoinsSpend_per_Round.mean()

