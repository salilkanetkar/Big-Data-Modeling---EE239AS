from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold

#This function performs the pre-processing on the data
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    filtered_words_final = [stemmer.stem(plural) for plural in filtered_words]
    filtered_words_final = [i for i in filtered_words_final if not i.isdigit()]
    return " ".join(filtered_words_final)

#This function plots the ROC
def roc(test,classifier_soft,gam,i):
    probas_ = classifier_soft.predict_proba(X_cv[test])                                    
    fpr, tpr, thresholds = roc_curve(cv_data.target[test], probas_[:, 1])
    s1 = 'SVM ROC for Gamma=%f' %  gam 
    s2 = ' & Fold=%d' % i
    s = s1+s2
    plt.plot(fpr, tpr, lw=1, label = s)                                    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc="lower right")
    plt.show()
    return

#This function varies the gamma everytime and generates a new SVM Model
def gamma_vary(gam):
    classifier_soft = svm.SVC(kernel='linear', probability=True,gamma = gam)
    print "*****************************************************************************************"
    print "The value of Gamma is %f" % gam
    for i, (train, test) in enumerate(cv):
        print "\nFold Number: %d" % (i+1)
        Y_test_predicted = classifier_soft.fit(X_cv[train], Y_cv[train]).predict(X_cv[test])
        accuracy = np.mean(Y_test_predicted == cv_data.target[test]) 
        print "The accuracy is %f" % accuracy
        print '\'0\' is Computer Technology and \'1\' is Recreational Activity'
        print "The precision and recall values are:"        
        print metrics.classification_report(Y_cv[test], Y_test_predicted)
        print 'The confusion matrix is as shown below:'
        print metrics.confusion_matrix(Y_cv[test], Y_test_predicted)
        print 'The ROC Curve is as shown below:'
        roc(test,classifier_soft,gam,i+1)

#Extracting the specified categories      
comp_rec = ['comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey']

#Fetching the data
cv_data = fetch_20newsgroups(subset='all',categories=comp_rec, remove=('headers','footers','quotes'))

#Performing pre-processing on the data
size, = cv_data.filenames.shape
for it in range(0,size):
    print it
    sentence = cv_data.data[it]
    cv_data.data[it] = preprocess(sentence)

#Generating the DTM Matrix
count_vect = CountVectorizer()
X_cv_counts = count_vect.fit_transform(cv_data.data)

#Generating the TF-IDF Matrix from DTM
tf_transformer = TfidfTransformer(use_idf=True).fit(X_cv_counts)
X_cv_tfidf = tf_transformer.transform(X_cv_counts)
docs,terms = X_cv_tfidf.shape

#Performing dimensionality reduction on the TF-IDF matrix
svd = TruncatedSVD(n_components=50, random_state=42)
X_cv = svd.fit_transform(X_cv_tfidf)
X_cv = Normalizer(copy=False).fit_transform(X_cv)
Y_cv = cv_data.target
for i in range(0,len(Y_cv)):
    if(Y_cv[i] <= 3):
        Y_cv[i] = 0
    else:
        Y_cv[i] = 1    

#Creating Folds
cv = StratifiedKFold(Y_cv, n_folds=5)

#Varying the values of Gamma
gamma_arr = [0.001,0.01,0.1,1,10,100,1000]
for i in range(0,len(gamma_arr)):
    gamma_vary(gamma_arr[i])