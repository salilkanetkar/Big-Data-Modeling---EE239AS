from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np

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

#The function sorts the tf-icf matrix to report the 10 most significant terms
def mySort(row, features):
    row = row[0]
    yx = zip(row, features)
    yx.sort()
    print yx[-9:]
    return

categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

#Fetching the dataset
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, remove=('headers','footers','quotes'))

#Preprocesses every document in the dataset
size, = twenty_train.filenames.shape
for it in range(0,size):
    print it
    sentence = twenty_train.data[it]
    twenty_train.data[it] = preprocess(sentence)

#Transferring the modified dataset into a Term Document Matrix     
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
docs,terms = X_train_counts.shape

#Creates an empty TF-ICF matrix
tf_icf = np.zeros(shape=(20,terms))

#Transfers the TDM into 20 categories TDM by adding values
for it in range(0,docs):
    print it
    cat = twenty_train.target[it]
    tf_icf[cat,] = tf_icf[cat,] + X_train_counts[it,]

#Calculates the TF-ICF for every category
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
tf_icf_final = tf_transformer.transform(tf_icf)
features = count_vect.get_feature_names()

print 'The 10 most significant terms in comp.sys.ibm.pc.hardware are:'
mySort(tf_icf_final[3].toarray(), features)
print 'The 10 most significant terms in comp.sys.mac.hardware are:'
mySort(tf_icf_final[4].toarray(), features)
print 'The 10 most significant terms in misc.forsale are:'
mySort(tf_icf_final[6].toarray(), features)
print 'The 10 most significant terms in soc.religion.christian are:'
mySort(tf_icf_final[15].toarray(), features)