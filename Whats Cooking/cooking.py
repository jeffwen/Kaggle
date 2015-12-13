path = '/Users/Jeffwen/Documents/Python/kaggle/Whats Cooking'

import json
import xgboost as xgb
import pandas as pd
import re
import numpy as np
import scipy
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from progressbar import ProgressBar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model, cross_validation, grid_search, svm
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer



# json reader
with open('train.json') as train_file,open('test.json') as test_file:
    data = json.load(train_file)
    test_set = json.load(test_file)

# json reader TEST TEST TEST
with open('weird.json') as weird_file:
    weird = json.load(weird_file)

    
# get training data
x_train = list([data[i]['ingredients'] for i in range(len(data))])
y_train = [data[i]['cuisine'] for i in range(len(data))]

#get test data
x_test = list([test_set[i]['ingredients'] for i in range(len(test_set))])
x_test_labels = [test_set[i]['id'] for i in range(len(test_set))]



# #------preprocessing ------#

# # return base form of each word in the ingredient lists
# def preprocess(input_data):
#     new_data = []
#     pbar = ProgressBar() # Progress bar to ease the waiting
#     for recipe in pbar(input_data):
#         new_recipe = []
#         for ingredient in recipe:
#             new_ingredient = []
#             for word in ingredient.split():
#                 word = re.sub('[^a-zA-Z -]+', '', word)
#                 new_ingredient.append(wn.morphy(word.lower().strip(",.!:?;' ")) or word.strip(",.!:?;' "))
#             new_recipe.append(' '.join(new_ingredient))
#         new_data.append(new_recipe)
#     return new_data

# # sort the words in the ingredients 
# def sort_ingredient(input_data):
#     new_data = []
#     pbar = ProgressBar()
#     for recipe in pbar(input_data):
#         new_recipe = []
#         for ingredient in recipe:
#             sorted_ingredient = ' '.join(sorted(ingredient.split(' ')))
#             new_recipe.append(sorted_ingredient)
#         new_data.append(new_recipe)
#     return new_data

# # count ingredients for each recipe
# def count_ingredient(input_data):
#     new_data = input_data.toarray()
#     aList = []
#     for row in new_data:
#         aList.append(len([x for x in row if x!=0.0]))
#     new_data = np.c_[new_data,aList]
#     return scipy.sparse.csr_matrix(new_data)


# # generate new training set                
# x_train_new = sort_ingredient(x_train)
# x_train_new = preprocess(x_train_new)

# # need to encode the training data
# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)

# # generate new test set
# x_test_new = preprocess(x_test)
# x_test_new = sort_ingredient(x_test_new)


# # defining new analyzer for tfidf (sklearn)reaks up the ingredients but this analyzer will treat the entire ingredient as one item
# # the original analyzer b
# def analyzer(aStr):
#     return aStr

# # training data tfidf
# tfidf = TfidfVectorizer(analyzer = analyzer,strip_accents='unicode')
# x_train_tfidf = tfidf.fit_transform(x_train_new)
# x_train_tfidf = count_ingredient(x_train_tfidf)
    
# # test data tfidf
# x_test_transformed = tfidf.transform(x_test_new)
# x_test_transformed = count_ingredient(x_test_transformed)


#------preprocessing 2------#
def preprocess_all_ingredients(input_data):
    new_data = []
    pbar = ProgressBar() # Progress bar to ease the waiting
    for recipe in pbar(input_data):
        new_recipe = []
        for ingredient in recipe:
            new_ingredient = []
            for word in ingredient.split():
                word = re.sub('[^a-zA-Z]', ' ', word)
                new_ingredient.append(WordNetLemmatizer().lemmatize(word.lower().strip(",.!:?;' ")) or word.strip(",.!:?;' "))
            new_recipe.append(' '.join(new_ingredient))
        new_data.append(' '.join(new_recipe))
    return new_data

# training data
x_train_new = preprocess_all_ingredients(x_train)

# test data
x_test_new = preprocess_all_ingredients(x_test)

# TF-IDF processing
tfidf_vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,1), analyzer='word', max_df=0.56, token_pattern=r'\w+')
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train_new)
x_test_tfidf = tfidf_vectorizer.transform(x_test_new)

#------Logistic regression------#

# fitting the model
lm = linear_model.LogisticRegression(solver='lbfgs',class_weight = 'balanced',multi_class='multinomial',C=5)
lm.fit(x_train_tfidf,y_train_encoded)

# running cross validation to check the model
scores = cross_validation.cross_val_score(linear_model.LogisticRegression(solver='lbfgs',class_weight = 'balanced',multi_class='multinomial',C=5), x_train_tfidf ,y_train_encoded, scoring='f1_weighted', cv=5)

# predicting test cuisines
lm_prediction = lm.predict(x_test_transformed)
test_Y_lm = label_encoder.inverse_transform(lm_prediction)

d = pd.DataFrame(data=OrderedDict([('id', x_test_labels), ('cuisine', test_Y_lm)]))
d.to_csv('submission_logreg_1.csv', index=False)


#------Logistic regression 2 Word lemmatization------#

# fitting the model
lm = linear_model.LogisticRegression(solver='lbfgs',class_weight = 'balanced',multi_class='multinomial',C=5)
lm.fit(x_train_tfidf,y_train)

# running cross validation to check the model
scores = cross_validation.cross_val_score(linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial',C=7), x_train_tfidf ,y_train, scoring='f1_weighted', cv=5)

# predicting test cuisines
lm_prediction_1 = lm.predict(x_test_tfidf)

d = pd.DataFrame(data=OrderedDict([('id', x_test_labels), ('cuisine', lm_prediction_1)]))
d.to_csv('submission_logreg_3.csv', index=False)

# grid search over regularization param
parameters = {'C':[1, 10]}
classifier = grid_search.GridSearchCV(lm, parameters)
classifier= classifier.fit(x_train_tfidf,y_train)
predictions= classifier.predict(x_test_tfidf)


#------linear SVM------#
svc = svm.LinearSVC(C=0.8, dual=False, penalty = 'l2')
svc.fit(x_train_tfidf,y_train)

# running cross validation to check the model
scores = cross_validation.cross_val_score(svm.LinearSVC(C=0.8, dual=False, penalty = 'l2'), x_train_tfidf ,y_train, scoring='f1_weighted', cv=5)

svc_prediction_1 = svc.predict(x_test_tfidf)

d = pd.DataFrame(data=OrderedDict([('id', x_test_labels), ('cuisine', svc_prediction_1)]))
d.to_csv('submission_svc_1.csv', index=False)


#------SVM------#

param_grid = [{'C': [0.1, 1, 10], 'kernel': ['linear']},{'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf']},]
svm_kern = svm.SVC()
clf = grid_search.GridSearchCV(svm_kern, param_grid)
clf.fit(x_train_tfidf, y_train)
svm_predictions= clf.predict(x_test_tfidf)

d = pd.DataFrame(data=OrderedDict([('id', x_test_labels), ('cuisine', svm_predictions)]))
d.to_csv('submission_svm_2.csv', index=False)


#------xgboost------#

sz = x_train_tfidf.shape

train_xgb = x_train_tfidf[:int(sz[0] * 0.8), :]
test_xgb= x_train_tfidf[int(sz[0] * 0.8):, :]

train_xgb_y= y_train_encoded[:int(sz[0] * 0.8),]
test_xgb_y = y_train_encoded[int(sz[0] * 0.8):,]

xgb_train = xgb.DMatrix(train_xgb, label = train_xgb_y)
xgb_test = xgb.DMatrix(test_xgb, label = test_xgb_y)

test_real = xgb.DMatrix(x_test_transformed)

# xgboost parameter setting
param = {'objective':'multi:softmax','num_class':20,'silent':1, 'eta':0.2, 'max_depth':20,'subsample':0.65}
watchlist = [(xgb_train,'train'), (xgb_test, 'test')]

# model training
num_round = 200
bst = xgb.train(param,xgb_train,num_round,watchlist)

# xgb prediction
pred = bst.predict(test_real)

pred_real = [int(i) for i in pred]
pred_real = label_encoder.inverse_transform(pred_real)

d = pd.DataFrame(data=OrderedDict([('id', x_test_labels), ('cuisine', pred_real)]))
d.to_csv('xgb_1.csv', index=False)


#------levenshtein------#
def levenshtein(s, t):
        ''' From Wikipedia article; Iterative with two matrix rows. '''
        if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j] 
        return v1[len(t)]

1 - (levenshtein(b[0],b[1])/float(max(len(b[0]),len(b[1]))))
