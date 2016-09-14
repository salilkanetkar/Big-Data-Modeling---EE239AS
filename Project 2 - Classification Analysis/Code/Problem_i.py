from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB    

#The function below performs the pre-processing and cleaning on the data
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    filtered_words_final = [stemmer.stem(plural) for plural in filtered_words]
    filtered_words_final = [i for i in filtered_words_final if not i.isdigit()]
    return " ".join(filtered_words_final)

#Extracting the specified categories   
cat_four=['comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'misc.forsale',
'soc.religion.christian']

#Fetching the training data
train_data = fetch_20newsgroups(subset='train',categories=cat_four, remove=('headers','footers','quotes'))
#Fetching the testing data
test_data =  fetch_20newsgroups(subset='test',categories=cat_four, remove=('headers','footers','quotes'))

#Preprocesses every document in the train dataset
size1, = train_data.filenames.shape
for it in range(0,size1):
    print it
    sentence = train_data.data[it]
    train_data.data[it] = preprocess(sentence)

#Preprocesses every document in the test dataset
size2, = test_data.filenames.shape
for it in range(0,size2):
    print it
    sentence = test_data.data[it]
    test_data.data[it] = preprocess(sentence)

#Transferring the modified train dataset into a Term Document Matrix    
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data.data)

#Calculating the TF-IDF values for every term in the document for train data
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts)
docs1,terms1 = X_train_tfidf.shape

#Performing dimensionality reduction using LSI and SVD on train data
svd = TruncatedSVD(n_components=50, random_state=42)
X_train = svd.fit_transform(X_train_tfidf)
X_train = Normalizer(copy=False).fit_transform(X_train)
Y_train = train_data.target

#Transferring the modified test dataset into a Term Document Matrix 
X_test_counts = count_vect.transform(test_data.data)

#Calculating the TF-IDF values for every term in the document for test data
tf_transformer = TfidfTransformer(use_idf=True).fit(X_test_counts)
X_test_tfidf = tf_transformer.transform(X_test_counts)
docs2,terms2 = X_test_tfidf.shape

#Performing dimensionality reduction using LSI and SVD on test data
X_test = svd.transform(X_test_tfidf)
X_test = Normalizer(copy=False).fit_transform(X_test)
Y_test = test_data.target

#Training and fitting the OneVsRestClassifier for Multiclass SVM
onevsrest = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train)
Y_pred1 = onevsrest.predict(X_test)

#Calculating the accuracy, recall, precision and confusion matrix
accuracy1 = np.mean(Y_test == Y_pred1)
print 'The accuracy for the model is %f' % accuracy1
print "The precision and recall values are:"
print metrics.classification_report(Y_test, Y_pred1)
print 'The confusion matrix is as shown below:'
print metrics.confusion_matrix(Y_test, Y_pred1)

#Training and fitting the OneVsOneClassifier for Multiclass SVM
onevsone = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train)
Y_pred2 = onevsone.predict(X_test)

#Calculating the accuracy, recall, precision and confusion matrix
accuracy2 = np.mean(Y_test == Y_pred2)
print 'The accuracy for the model is %f' % accuracy2
print "The precision and recall values are:"
print metrics.classification_report(Y_test, Y_pred2)
print 'The confusion matrix is as shown below:'
print metrics.confusion_matrix(Y_test, Y_pred2)

#Training and Fitting the Multi class Naive Bayes Model with OneVsRestClassifier
onevsrest = OneVsRestClassifier(GaussianNB()).fit(X_train, Y_train)
Y_pred3 = onevsrest.predict(X_test)

#Calculating the accuracy, recall, precision and confusion matrix
accuracy3 = np.mean(Y_test == Y_pred3)
print 'The accuracy for the model is %f' % accuracy3
print "The precision and recall values are:"
print metrics.classification_report(Y_test, Y_pred3)
print 'The confusion matrix is as shown below:'
print metrics.confusion_matrix(Y_test, Y_pred3)

#Training and Fitting the Multi class Naive Bayes Model with OneVsOneClassifier
onevsone = OneVsOneClassifier(GaussianNB()).fit(X_train, Y_train)
Y_pred4 = onevsone.predict(X_test)

#Calculating the accuracy, recall, precision and confusion matrix
accuracy4 = np.mean(Y_test == Y_pred4)
print 'The accuracy for the model is %f' % accuracy4
print "The precision and recall values are:"
print metrics.classification_report(Y_test, Y_pred4)
print 'The confusion matrix is as shown below:'
print metrics.confusion_matrix(Y_test, Y_pred4)