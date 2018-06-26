from skmultiflow.classification.lazy.knn import KNN
from skmultiflow.data.file_stream import FileStream
import time
import pudb;pu.db
f="/opt/techgig/scikit-multiflow/src/skmultiflow/datasets/sea_big.csv"
stream = FileStream(f, -1, 1)
stream.prepare_for_use()
X, y = stream.next_sample(200)
knn = KNN(k=8, max_window_size=2000, leaf_size=40)
knn.partial_fit(X, y)
n_samples = 0
corrects = 0
while n_samples < 5000:
    X, y = stream.next_sample()
    my_pred = knn.predict(X)
    if y[0] == my_pred[0]:
        corrects += 1
    knn = knn.partial_fit(X, y)
    n_samples += 1
    print("KNN's performance: " + str(corrects/n_samples))
    time.sleep(1)
print('KNN usage example')
print(str(n_samples) + ' samples analyzed.')
print("KNN's performance: " + str(corrects/n_samples))
