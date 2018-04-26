
# coding: utf-8

# In[1]:


#nltk.download()


# In[2]:


import os
from collections import Counter
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def make_Dictionary(root_dir):
    emails= [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for line in m:
                words = line.split()
                all_words += words
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary)

    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif item in stopwords.words('english'):
            del dictionary[item]
        elif item in ENGLISH_STOP_WORDS:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    #print(dictionary)

    return dictionary


def extract_features(root_dir):
    emails= [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    docID = 0
    features_matrix = np.zeros((1000, 3000))
    train_labels = np.zeros(1000)
    for mail in emails:
        with open(mail) as m:
            all_words = []
            for line in m:
                words = line.split()
                all_words += words
            for word in all_words:
                wordID = 0
                for i, d in enumerate(dictionary):
                    if d[0] == word:
                        wordID = i
                        features_matrix[docID, wordID] = all_words.count(word)
        train_labels[docID] = int(mail.split(".")[-2] == 'spam')
        docID = docID + 1
    return features_matrix, train_labels


# Create a dictionary of words with its frequency
print("start")
root_dir = "spam_1000"
dictionary = make_Dictionary(root_dir)

# Prepare feature vectors per training mail and its labels

features_matrix, labels = extract_features(root_dir)


# In[6]:

#print(features_matrix.shape)
#print(labels.shape)
#print(sum(labels == 0), sum(labels == 1))


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)

## Training models and its variants
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier

model1 = LinearSVC()
model2 = MultinomialNB()
model3 = DecisionTreeClassifier()
model4 = LogisticRegression()
model5 = RandomForestClassifier(max_depth=10, random_state=0)
model6 = AdaBoostClassifier()
model7 = BaggingClassifier()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)
model6.fit(X_train, y_train)
model7.fit(X_train, y_train)

result1 = model1.predict(X_test)
result2 = model2.predict(X_test)
result3 = model3.predict(X_test)
result4 = model4.predict(X_test)
result5 = model5.predict(X_test)
result6 = model6.predict(X_test)
result7 = model7.predict(X_test)

#print (confusion_matrix(y_test, result1))
#print (confusion_matrix(y_test, result2))
#print (confusion_matrix(y_test, result3))

import pandas as pd
d = {'List1' : pd.Series(result1),'List2' : pd.Series(result2),'List3': pd.Series(result3),'List4': pd.Series(result4),'List5': pd.Series(result5),'List6': pd.Series(result6),'List7': pd.Series(result7)}

df = pd.DataFrame(d)

ensemble_result=[]

for i in range(len(df)):
    sum=df['List1'][i]+df['List2'][i]+df['List3'][i]+df['List4'][i]+df['List5'][i]+df['List6'][i]+df['List7'][i]
    if sum>3:
        ensemble_result.append(1)
    else:
        ensemble_result.append(0)
d2 = {'LinearSVC' : pd.Series(result1),'MultinomialNB' : pd.Series(result2),'DecisionTree': pd.Series(result3),'LogisticRegression': pd.Series(result4),'RandomForest': pd.Series(result5),'AdaBoost': pd.Series(result6),'Bagging': pd.Series(result7),'ensemble': pd.Series(ensemble_result)}    
df2 = pd.DataFrame(d2)
print(df2)


# In[12]:


print (confusion_matrix(y_test, result1))
print (confusion_matrix(y_test, result2))
print (confusion_matrix(y_test, result3))
print (confusion_matrix(y_test, result4))
print (confusion_matrix(y_test, result5))
print (confusion_matrix(y_test, result6))
print (confusion_matrix(y_test, result7))
print (confusion_matrix(y_test, ensemble_result))


# In[ ]:

'''
X_train, X_test, y_train, y_test


# In[12]:


tn, fp, fn, tp = confusion_matrix(y_test, result1).ravel()
(tn, fp, fn, tp)


# In[13]:


tn, fp, fn, tp = confusion_matrix(y_test, result2).ravel()
(tn, fp, fn, tp)


# In[9]:


tn, fp, fn, tp = confusion_matrix(y_test, result3).ravel()
(tn, fp, fn, tp)'''


# In[9]:




def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[15]:


plot_confusion_matrix(cm           = np.array(confusion_matrix(y_test, result1)), 
                      normalize    = False,
                      target_names = ['ham', 'spam'],
                      title        = "SVM Confusion Matrix")

plot_confusion_matrix(cm           = np.array(confusion_matrix(y_test, result2)), 
                      normalize    = False,
                      target_names = ['ham', 'spam'],
                      title        = "naive_bayes Confusion Matrix")

plot_confusion_matrix(cm           = np.array(confusion_matrix(y_test, result3)), 
                      normalize    = False,
                      target_names = ['ham', 'spam'],
                      title        = "DecisionTree Classifier Confusion Matrix")
plot_confusion_matrix(cm           = np.array(confusion_matrix(y_test, result4)), 
                      normalize    = False,
                      target_names = ['ham', 'spam'],
                      title        = "Logistic Regression Confusion Matrix")
plot_confusion_matrix(cm           = np.array(confusion_matrix(y_test, result5)), 
                      normalize    = False,
                      target_names = ['ham', 'spam'],
                      title        = "random forest Confusion Matrix")
plot_confusion_matrix(cm           = np.array(confusion_matrix(y_test, result6)), 
                      normalize    = False,
                      target_names = ['ham', 'spam'],
                      title        = "AdaBoost Classifier Confusion Matrix")
plot_confusion_matrix(cm           = np.array(confusion_matrix(y_test, result7)), 
                      normalize    = False,
                      target_names = ['ham', 'spam'],
                      title        = "Bagging Classifier Confusion Matrix")
plot_confusion_matrix(cm           = np.array(confusion_matrix(y_test, ensemble_result)), 
                      normalize    = False,
                      target_names = ['ham', 'spam'],
                      title        = "ensemble Confusion Matrix")

