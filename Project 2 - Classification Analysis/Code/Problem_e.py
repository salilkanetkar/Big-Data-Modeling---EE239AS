from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt

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
comp_rec = ['comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey']
 
#Fetching the training data
train_data = fetch_20newsgroups(subset='train',categories=comp_rec, remove=('headers','footers','quotes'))
#Fetching the testing data
test_data =  fetch_20newsgroups(subset='test',categories=comp_rec, remove=('headers','footers','quotes'))

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
for i in range(0,len(Y_train)):
    if(Y_train[i] <= 3):
        Y_train[i] = 0
    else:
        Y_train[i] = 1    

#Transferring the modified test dataset into a Term Document Matrix 
X_test_counts = count_vect.transform(test_data.data)

#Calculating the TF-IDF values for every term in the document for test data
tf_transformer = TfidfTransformer(use_idf=True).fit(X_test_counts)
X_test_tfidf = tf_transformer.transform(X_test_counts)
docs2,terms2 = X_test_tfidf.shape

#Performing dimensionality reduction using LSI and SVD on test data
X_test = svd.transform(X_test_tfidf)
X_test = Normalizer(copy=False).fit_transform(X_test)

shape, = test_data.target.shape
for i in range(0,shape):
    if(test_data.target[i] <= 3):
        test_data.target[i] = 0
    else:
        test_data.target[i] = 1    

#Training the SVM Classifier
classifier = svm.SVC(kernel='linear', probability=True)
model = classifier.fit(X_train, Y_train)

#Prediciting on the test data using the tarined model
Y_test_predicted = model.predict(X_test)

#Calculating the accuracy, recall, precision and confusion matrix
accuracy_svm = np.mean(test_data.target == Y_test_predicted)
print 'The accuracy for the model is %f' % accuracy_svm
print '\'0\' is Computer Technology and \'1\' is Recreational Activity'
print "The precision and recall values are:"
print metrics.classification_report(test_data.target, Y_test_predicted)
print 'The confusion matrix is as shown below:'
print metrics.confusion_matrix(test_data.target, Y_test_predicted)

#Plotting the ROC
probas_ = model.predict_proba(X_test)                                    
fpr, tpr, thresholds = roc_curve(test_data.target, probas_[:, 1])
plt.plot(fpr, tpr, lw=1, label = "SVM ROC")                                    
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")
plt.show()