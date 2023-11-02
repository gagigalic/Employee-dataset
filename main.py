import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Employee.csv")
print(df.head())
print(df.describe())
print(df.info())

#all columns name
columns = df.columns
print(columns)

#datatype of each column
type = df.dtypes
print(columns)

#number of unique value of all columns
unique_value  = df.nunique()
print(unique_value)

#check Is there any null values in the dataset
print(df.isnull().sum())
sns.heatmap(df.isnull())
plt.savefig("heatmap.png")
plt.close()
#no null values

#shape of data
shape = df.shape
print(shape)

#number of unique degrees
print(df["Education"].unique())

deg = df["Education"].value_counts().reset_index()
print(deg)

plt.bar(df.Education.unique(), df.Education.value_counts(),color='purple')
plt.savefig("Education.png")
plt.close()

plt.pie(x=df['Education'].value_counts(),labels=df['Education'].value_counts().index,autopct='%1.1f%%',explode=[0,0.1,0.1])
plt.title("Different Degree Holder")
plt.savefig("Different_degree.png")
plt.close()

#number of unique city
print(df["City"].unique())

city=df['City'].value_counts().reset_index()
print(city)

plt.bar(df['City'].unique(),df.City.value_counts(),color='red')
plt.savefig("City.png")
plt.close()

plt.pie(x=df['City'].value_counts(),labels=df['City'].value_counts().index,autopct='%1.1f%%',explode=[0,0.1,0.1])
plt.title("Employee from different City")
plt.savefig("Different_city.png")
plt.close()

#Payment Tier
print(df["PaymentTier"].unique())

pay_tier=df['PaymentTier'].value_counts().reset_index()
print(pay_tier)

plt.bar(df.PaymentTier.unique(),df.PaymentTier.value_counts(),color='pink')
plt.savefig("PaymentTier.png")
plt.close()

plt.pie(x=df.PaymentTier.value_counts(),labels=df.PaymentTier.value_counts().index,autopct="%1.1f%%",explode=[0,0.1,0.1])
plt.title("Employees from different PaymentTier")
plt.savefig("Different_paymentTier.png")
plt.close()


#LeaveOrNot
print(df["LeaveOrNot"].unique())

leave=df['LeaveOrNot'].value_counts().reset_index()
print(leave)

plt.bar(df.LeaveOrNot.unique(),df.LeaveOrNot.value_counts(),color='blue')
plt.savefig("LeaveOrNot.png")
plt.close()

plt.pie(x=df.LeaveOrNot.value_counts(),labels=df.LeaveOrNot.value_counts().index,autopct="%1.1f%%",explode=[0,0.1])
plt.title("Rate of LeaveOrNot")
plt.savefig("Rate_leave_or_not.png")
plt.close()

#age
print(df['Age'].unique())

age=df['Age'].value_counts().reset_index()
age=age.sort_values(by='Age')
print(age)

plt.bar(df.Age.unique(),df.Age.value_counts(),color='black')
plt.savefig("Age.png")
plt.close()

plt.pie(x=df.Age.value_counts(),labels=df.Age.value_counts().index,autopct="%1.1f%%")
plt.title("Ratio Of Different Ages of Employees")
plt.savefig("Different_age.png")
plt.close()

#Gender
print(df['Gender'].unique())

gender=df['Gender'].value_counts().reset_index()
print(gender)

plt.bar(df.Gender.unique(),df.Gender.value_counts())
plt.savefig("Gender.png")
plt.close()

plt.pie(x=df.Gender.value_counts(),labels=df.Gender.value_counts().index,autopct="%0.1f%%",explode=[0,0.09])
plt.title("Gender Ratio")
plt.savefig("Gender_ratio.png")
plt.close()

#Ever Benched
print(df['EverBenched'].unique())

ever_benched=df['EverBenched'].value_counts().reset_index()
print(ever_benched)

plt.bar(df.EverBenched.unique(),df.EverBenched.value_counts())
plt.savefig("EverBenched.png")
plt.close()

plt.pie(x=df.EverBenched.value_counts(),labels=df.EverBenched.value_counts().index,autopct="%1.1f%%",explode=[0,0.09])
plt.title("Ratio of EverBenched")
plt.savefig("Ratio_of_everBenched.png")
plt.close()

#Experience
print(df['ExperienceInCurrentDomain'].unique())

experience=df['ExperienceInCurrentDomain'].value_counts().reset_index()
print(experience)

plt.bar(df.ExperienceInCurrentDomain.unique(),df.ExperienceInCurrentDomain.value_counts(),color='brown')
plt.savefig("ExperienceInCurrentDomain.png")
plt.close()

plt.pie(x=df.ExperienceInCurrentDomain.value_counts(),labels=df.ExperienceInCurrentDomain.value_counts().index,autopct="%1.1f%%")
plt.title("Experience Ratio In Current Domain")
plt.savefig("Experience_Ratio.png")
plt.close()

#Bivariate Analsysis

sns.countplot(x='JoiningYear',hue='City',data=df,)
plt.xlabel("Joining Year")
plt.ylabel("coun of new employee")
plt.title("New Joiners across different cities")
plt.savefig("New_joiners.png")
plt.close()
#Bangalore has the most hiring over the years


#Label Encoding for Categorical Columns

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['Education']=le.fit_transform(df['Education'])
df['City']=le.fit_transform(df.City)
df['Gender']=le.fit_transform(df['Gender'])
df['EverBenched']=le.fit_transform(df['EverBenched'])

print(df.head(10))

target=df['LeaveOrNot']
X=df.drop(['LeaveOrNot'],axis='columns')
y=target

#co-relation between different columns
corr = print(df.corr())

plt.figure(figsize=(12,7))
sns.heatmap(df.corr(), annot = True, cmap = "viridis")
plt.savefig("Corr.png")
plt.close()
# Education, JoiningYear, City, EverBenched affect the LeaveOrNot values affect the LeaveOrNot values


#Education Vs LeaveOrNot
sns.countplot(x='Education',hue='LeaveOrNot',data=df)
plt.savefig("plot.png")
plt.close()
#Bachelors" 1="Masters" 2="P.hd",employees who have bachelor's Degree have low chances of leaving & the employees who have master's degree are more likely to quit the job

#Joining Year Vs LeaveOrNot
sns.countplot(x='JoiningYear',hue='LeaveOrNot',data=df)
plt.savefig("plot2.png")
plt.close()
# in 2018 people who joined are the most to take the leaves

#City Vs LeaveOrNot
sns.countplot(x='City',hue='LeaveOrNot',data=df)
plt.savefig("plot3.png")
plt.close()
#'0'=Bangalore, '1'=New Delhi, '2'=Pune,employee which are from Bangalore are less likely to leave the job as compared to New Delhi and Pune

#Age Vs LeaveOrNot
sns.countplot(x='Age',hue='LeaveOrNot',data=df)
plt.savefig("plot4.png")
plt.close()

sns.countplot(x = 'Age' , hue = 'LeaveOrNot', data = df.loc[df.Education == 1])
plt.savefig("plot5.png")
plt.close()
#employee who have master's degree and are in between 24 - 27 are more likely to leave the job

#PaymentTier Vs LeaveOrNot
sns.countplot(x="PaymentTier",hue="LeaveOrNot",data=df)
plt.savefig("plot6.png")
plt.close()
#employee who recieve payment of 2nd tier leave the company more as compared to employees who get 1 & 3 tier paymen

#Gender Vs LeaveOrNot
sns.countplot(x='Gender',hue='LeaveOrNot',data=df)
plt.savefig("plot7.png")
plt.close()
#'0'=female '1'=male,female are majority on leave, despite being less number than men

#Split the Data into Train and Test Set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Applying Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

lr=LogisticRegression(max_iter=4000)
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))

predictions = lr.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
#Logistic Regression gives 72% of accuracy

#Applying Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
print(rfc.score(X_test,y_test))

predictions2 = rfc.predict(X_test)
print(classification_report(y_test,predictions2))
print(confusion_matrix(y_test,predictions2))
#Random Forest gives 85% of accuracy

#Applying Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
print(dtc.score(X_test,y_test))

predictions3 = dtc.predict(X_test)
print(classification_report(y_test,predictions3))
print(confusion_matrix(y_test,predictions3))
#DecisionTree gives 83% of accuracy

#Applying Support Vector Machine
from sklearn import svm

sv = svm.SVC(C = 40, kernel = 'rbf')
sv.fit(X_train,y_train)

print(sv.score(X_test,y_test))

predictions4 = sv.predict(X_test)
print(classification_report(y_test,predictions4))
print(confusion_matrix(y_test,predictions4))
#Support Vector gives 66% of accuracy

#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()
gnb.fit(X_train,y_train)
print(gnb.score(X_test,y_test))

predictions5 = gnb.predict(X_test)
print(classification_report(y_test,predictions5))
print(confusion_matrix(y_test,predictions5))
#Gaussian gives 68% of accuracy

#Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB

mnb=MultinomialNB()
mnb.fit(X_train,y_train)
print(mnb.score(X_test,y_test))

predictions6 = mnb.predict(X_test)
print(classification_report(y_test,predictions6))
print(confusion_matrix(y_test,predictions6))
#Multinomial gives 69% of accuracy

#The best model in RANDOM FOREST














