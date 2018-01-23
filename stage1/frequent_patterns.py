##Stage 1: Mining frequent patterns and association rules from the given data set.

import pandas as pd
import numpy as np
import re
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import OnehotTransactions

def find_years(name1, name2, authors):
    year = []
    for i in range(len(authors)):
        p1 = re.compile(name1)
        p2 = re.compile(name2)
        if p1.search(authors[i]) and p2.search(authors[i]):
            year.append([paperDataset.iloc[i].year,
                         paperDataset.iloc[i].paper_id])
    return year


def clean_name(dirty_name):
    s = str(dirty_name)
    return (s[s.find("'") + 1:s.rfind("'")])



# read paper_dataset, paramaters are: paper_id	venue	authors	year	title	index_keys	author_keys	abstract
paperDataset = pd.read_csv('paper_dataset.csv',encoding = "ISO-8859-1")

authors = paperDataset.authors
# print(authors)

author_list = []
for i in range(len(authors)):
    author_list.append(re.split(r'\s*\+\s*',authors[i]))
print("###############")
# print(author_list)

oht = OnehotTransactions()
oht_ary = oht.fit(author_list).transform(author_list)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
frequent_itemsets = apriori(df, min_support=0.006, use_colnames=True)
print("frequent_itemsets")
sorted_frequent_itemmsets = frequent_itemsets.sort_values(
    ["support"], ascending=False)
sorted_frequent_itemmsets.to_csv("sorted_frequent_itemmsets.csv")
print(sorted_frequent_itemmsets)


rules = association_rules(sorted_frequent_itemmsets,
                          metric="lift", min_threshold=1)
sorted_rules = rules.sort_values(["support"], ascending=False)
print(sorted_rules)

###print associated authores and their paper id 
for i in range(len(sorted_rules.antecedants)):
    name1 = clean_name(rules.iloc[i].antecedants)
    name2 = clean_name(rules.iloc[i].consequents)
    year = find_years(name1, name2, authors)
    print(name1 + " and " + name2 + ":", year)

###print the antecedants and consequents
# for i in range(len(sorted_rules)):
#     sorted_rules.set_value(i, 'antecedants', clean_name(
#         sorted_rules.get_value(i, 'antecedants')))
#     sorted_rules.set_value(i, 'consequents', clean_name(
#         sorted_rules.get_value(i, 'consequents')))
#     print(sorted_rules.get_value(i, 'antecedants'))
# sorted_rules.to_csv("sorted_rules.csv")
