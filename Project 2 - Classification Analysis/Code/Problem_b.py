from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

#The function below performs the pre-processing and cleaning on the data
def preprocess(sentence):
    sentence = sentence.lower() #transfers each word to lower case
    tokenizer = RegexpTokenizer(r'\w+') #Tokenizer to remove punctuation marks
    stemmer = PorterStemmer() #Stemmer to perform stemming
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')] #Removes stop words
    filtered_words_final = [stemmer.stem(plural) for plural in filtered_words]
    filtered_words_final = [i for i in filtered_words_final if not i.isdigit()] #Removes numbers and digits
    return " ".join(filtered_words_final)

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

#Fetching the data set
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, remove=('headers','footers','quotes'))

#Stores the size of the dataset
size, = twenty_train.filenames.shape

#Performing preprocessing on every document
for it in range(0,size):
    print it
    sentence = twenty_train.data[it]
    twenty_train.data[it] = preprocess(sentence)

#Transferring the modified dataset into a Term Document Matrix    
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

#Calculating the TF-IDF values for every term in the document
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts)
docs,terms = X_train_tfidf.shape

#Reporting the final number of terms
print "The final number of terms are", terms