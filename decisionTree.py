import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image, display
import graphviz
import numpy as np
import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
	df = pd.read_csv("D:/KrishnaBackup/Python Test/ML/Class/DecisionTree/train.csv")
	mragefill = df[df.Name.str.match(r'.*\bMr\b.*')].Age.dropna().mean()
	mrsagefill = df[df.Name.str.match(r'.*\bMrs\b.*')].Age.dropna().mean()
	masteragefill = df[df.Name.str.match(r'.*\bMaster\b.*')].Age.dropna().mean()
	missagefill = df[df.Name.str.match(r'.*\bMiss\b.*')].Age.dropna().mean()


	df.loc[df.Name.str.match(r'.*\bMr\b.*') & df.Age.isnull(),'Age']=mragefill
	df.loc[df.Name.str.match(r'.*\bMrs\b.*') & df.Age.isnull(),'Age']=mrsagefill
	df.loc[df.Name.str.match(r'.*\bMaster\b.*') & df.Age.isnull(),'Age']=masteragefill
	df.loc[df.Name.str.match(r'.*\bMiss\b.*') & df.Age.isnull(),'Age']=missagefill
	df = df.fillna(0)
	df['ticket_id'] = pd.factorize(df.Ticket)[0]
	df['sex_id'] = pd.factorize(df.Sex)[0]
	train_data = df[['sex_id','Age','Pclass']]
	train_data.head()
	
	train_target = df[['Survived']]
	train_target.head()
	
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(train_data, train_target)
	dot_data = tree.export_graphviz(clf, out_file=None)
	graph = graphviz.Source(dot_data)
	graph.render("titanic")
	
	# clf = tree.DecisionTreeClassifier()
	# clf = clf.fit(iris.data, iris.target)
	# dot_data = tree.export_graphviz(clf, out_file=None)
	# graph = graphviz.Source(dot_data)
	# graph.render("iris")
