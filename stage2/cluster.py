# Stage 2: Applying different clustering methods to the given data set.


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# read paper_dataset, paramaters are: paper_id	venue	authors	year	title	index_keys	author_keys	abstract
paperDataset = pd.read_csv('paper_dataset.csv', encoding="ISO-8859-1")

index_keys = paperDataset.index_keys
paper_id = paperDataset.paper_id

##############################    tfidf   ##########################################
countVect = CountVectorizer(stop_words="english", max_df=0.8)
vari_mat = countVect.fit_transform(index_keys)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vari_mat)

###  combine the cluser result and generate a dictionary, in which key is the cluster number and value is the paper id 
def combine_clusterResult(result, value):
	result_dic = {}
	for i in range(len(result)):
		temp = result[i]
		if temp in result_dic:
			result_dic[temp].append(value[i])
		else:
			result_dic[temp] = [value[i]]
	return (result_dic)


import matplotlib.pyplot as plt

def figure(input, str):
    index = []
    numbers = []
    for i, j in input.items():
        index.append(i)
        numbers.append(len(j))
    plt.figure('fig1')
    plt.bar(index, numbers, width=0.3, color="g",
            align="center", label="numbers")
    plt.xlabel(str + " clustering")
    plt.ylabel("numbers")
    plt.legend()
    plt.title("barplot")
    plt.show()

###############################   Kmeans clustering    ##########################################
from sklearn.cluster import KMeans

num_clusters = 10
km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=5,
                        init='k-means++', n_jobs=-1)
result = km_cluster.fit_predict(tfidf)
print(result)

result_dic=combine_clusterResult(result, paper_id)
print( result_dic)
################################   GAAC clustering    #####################################################
from nltk.cluster.gaac import GAAClusterer

num_clusters=10
gaac=GAAClusterer(num_clusters=num_clusters)
gaac.cluster(tfidf.toarray())
# gaac.dendrogram().show()
result = [gaac.classify(i) for i in tfidf.toarray()]
print ("Predicting result: ")
print(result)
result_dic=combine_clusterResult(result,paper_id)
print( result_dic)
figure(result_dic,"GAAC")






