import pandas
import numpy
import scipy
import sklearn
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, cross_validation , svm

names=['serial','timestamp','temp','humidity','light','co2','humidity','occupancy']
url="File:///E:/Projects Res/ML/occupancy_data/datatraining.data"
df=pandas.read_csv(url,names=names)

x=numpy.array(df.drop(['serial','timestamp','occupancy'],axis=1))
y=numpy.array(df['occupancy'])

x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=1)

clf=MultinomialNB()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test)*100,"%")
print("##############################################")

df2=pandas.read_csv("File:///E:/Projects Res/ML/occupancy_data/datatest2.txt",names=['serial','timestamp','temp','humidity','light','co2','humidity','occupancy'])
x1=numpy.array(df2.drop(['serial','timestamp','occupancy'],axis=1))
y1=numpy.array(df2['occupancy'])
print(clf.score(x1,y1)*100,"%")
ans=clf.predict(x1)
print(ans)
print(y1)
numpy.savetxt('E:/Projects Res/ML/occupancy_data/output_file_1.data', ans, delimiter=",", fmt="%s")
print("###############################################")

df3=pandas.read_csv("File:///E:Projects Res/ML/occupancy_data/datatest.txt",names=['serial','timestamp','temp','humidity','light','co2','humidity','occupancy'])
x2=numpy.array(df3.drop(['serial','timestamp','occupancy'],axis=1))
y2=numpy.array(df3['occupancy'])
print(clf.score(x2,y2)*100,"%")
ans=clf.predict(x2)
numpy.savetxt('E:/Projects Res/ML/occupancy_data/output_file_2.data', ans, delimiter=",", fmt="%s")
print(ans)
print(y2)

#df2.hist()
#plt.show()