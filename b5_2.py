from skmultiflow.classification.lazy.knn import KNN
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.data_stream import DataStream
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import time
import pudb;pu.db
f1="/opt/techgig/scikit-multiflow/src/skmultiflow/datasets/sea_big.csv"
f2="d1.csv"

df1 = pd.read_csv(f2)
#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#data = tfidf.fit_transform(df1.Title)
cv = CountVectorizer(stop_words='english')
data = cv.fit_transform(df1.Title)
dataM = data.toarray()
df2 = pd.DataFrame(dataM)
df2['Resolution'] = df1['Resolution']
df = df2
#df = pd.merge(df1.assign(A=1), df2.assign(A=1), on='A').drop('A', 1)
#df['Title']=data

#stream = FileStream(f, -1, 1)
stream = DataStream(df)
stream.prepare_for_use()
X, y = stream.next_sample(10)
knn = KNN(k=5, max_window_size=2000, leaf_size=40)
knn.partial_fit(X, y)
n_samples = 0
corrects = 0
while n_samples < 38:
    X, y = stream.next_sample()
    my_pred = knn.predict(X)
    if y[0] == my_pred[0]:
        corrects += 1
    knn = knn.partial_fit(X, y)
    n_samples += 1
    #print("KNN's performance: " + str(corrects/n_samples))
#    time.sleep(1)
print('KNN usage example')
print(str(n_samples) + ' samples analyzed.')
print("KNN's performance: " + str(corrects/n_samples))
