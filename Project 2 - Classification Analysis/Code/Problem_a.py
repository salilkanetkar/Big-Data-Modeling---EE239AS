import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.datasets import fetch_20newsgroups
py.sign_in('salil1993', 'nynwcmfwlg')

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
 
#Fetching the dataset from the package 
twenty_train = fetch_20newsgroups(subset='all',categories=categories)

#Storing the size of each category in the array below
size_arr = []
for it in range(0,len(categories)):
    twenty_train = fetch_20newsgroups(subset='all',categories=categories[it].split())
    size, = twenty_train.target.shape
    size_arr.append(size)

#Plotting the histogram
data = [
    go.Bar(
        x=categories,
        y=size_arr
    )
]
plot_url = py.plot(data, filename='basic-bar')
#The histogram opens up in a browser

#Calculating the count of 'Computer Technology'
comp = 0
for it in range(1,5):
    comp = comp + size_arr[it]
print 'The number of documents in Computer Technology are %d' % comp

#Calculating the count of 'Recreational Activity'
rec = 0
for it in range(7,11):
    rec = rec + size_arr[it]
print 'The number of documents in Recreational Activity are %d' % rec