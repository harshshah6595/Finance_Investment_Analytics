import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import lightgbm as lgb
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

df= pd.read_csv('training.csv')
df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df1=df.drop([ 'address_line_2','street_name','street_number','address_line_2','city_name','zip_code','build_date','remodel_date','misc_features','average_neighborhood_price','damage_code','floor_of_unit','bedrooms','bathrooms'] , axis=1)
sns.heatmap(df1.isnull(),yticklabels=False,cbar=False,cmap='viridis')

df1.isnull().values.any()
df1 = df1[np.isfinite(df1['floors_in_building'])]

#fill null values
mean= df1['schools_in_area'].median(skipna=True)
df1['schools_in_area']=df1['schools_in_area'].fillna(mean)
mean= df1['crime_score'].median(skipna=True)
df1['crime_score']=df1['crime_score'].fillna(mean)
mean= df1['culture_score'].median(skipna=True)
df1['culture_score']=df1['culture_score'].fillna(mean)
mean= df1['public_transit_score'].median(skipna=True)
df1['public_transit_score']=df1['public_transit_score'].fillna(mean)
mean= df1['schools_in_area'].median(skipna=True)
df1['overall_inspector_score']=df1['overall_inspector_score'].fillna(mean)
mean= df1['overall_inspector_score'].median(skipna=True)
df1['sqft']=df1['sqft'].fillna(mean)
mean= df1['sqft'].median(skipna=True)

df1['profit']=df1['final_price']-df1['investment']-df1['initial_price']
df1.loc[df1.profit < 500, 'purchase_decision'] = 0
df1['population_increase']=df1['current_population']-df1['population_5_years_ago']
df1.loc[df1.population_increase<0,'population_increase']=0
df1.loc[df1.population_increase>0,'population_increase']=1

sns.countplot(x='area_type',hue='purchase_decision',data=df1,palette='RdBu_r') 
#rural area have less ratio of success
sns.countplot(x='population_increase',hue='purchase_decision',data=df1,palette='RdBu_r') 
#if there is increase in population, demand increase and so profit increase
sns.countplot(x='structural_quality_grade',hue='purchase_decision',data=df1,palette='RdBu_r')
sns.countplot(x='utilities_grade',hue='purchase_decision',data=df1,palette='RdBu_r')
sns.countplot(x='interior_condition_grade',hue='purchase_decision',data=df1,palette='RdBu_r')
sns.countplot(x='exterior_condition_grade',hue='purchase_decision',data=df1,palette='RdBu_r')
sns.countplot(x='zone',hue='purchase_decision',data=df1,palette='RdBu_r')
sns.barplot(x='zone',y='profit',data=df1)
sns.barplot(x='area_type',y='final_price',data=df1)
cor=df1[['zone','days_on_market','schools_in_area','public_transit_score','current_population','population_5_years_ago','profit','crime_score','purchase_decision']]
sns.heatmap(cor.corr(),cmap='coolwarm',annot=True)

df2 = pd.get_dummies(df1, columns=['zone','sub_type','area_type','inspection_type','structural_quality_grade','exterior_condition_grade','interior_condition_grade','utilities_grade','exterior_color','exterior_material','damage_and_issue_grade'])
from sklearn.model_selection import train_test_split
X = df2.drop(['property_id','purchase_decision','population_5_years_ago', 'investment',
       'final_price', 'profit'],axis=1)
y = df2['purchase_decision']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
clf1 = LogisticRegression(random_state=1)
clf2= GradientBoostingClassifier(random_state=0)
clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('gbc', clf2), ('gnb', clf3)], voting='soft')
eclf1 = eclf1.fit(X_train, y_train)
pre=eclf1.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(pre,y_test))
print(confusion_matrix(y_test,pre))

X = df2.drop(['property_id','current_population','purchase_decision','population_5_years_ago', 'investment',
       'final_price', 'profit'],axis=1)
y = df2['profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.linear_model import LinearRegression
lm = LinearRegression(n_jobs=11)
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions,s=20)




