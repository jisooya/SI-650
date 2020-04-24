import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
TRAINING_DATA = "../train.tsv"
TESTING_DATA = "../test.tsv"
OUT_FILE = "../result.csv"
train_data = pd.read_csv(TRAINING_DATA, sep='\t')
labels = list(train_data['label'])
texts = list(train_data['text'])
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
# ngram level tf-idf
vec = TfidfVectorizer(ngram_range=(1, 3))
train_matrix = vec.fit_transform(trainDF['text'])
# train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['la
print("Done")
test_data = pd.read_csv(TESTING_DATA, sep='\t')
test_texts = test_data['text']
test_matrix = vec.transform(test_texts)

clf = svm.LinearSVC(C=2.0) clf.fit(train_matrix, labels) predicted_labels = clf.predict(test_matrix)
# print(predicted_labels)
# clf = LogisticRegression(penalty="l2")
# clf.fit(train_matrix, labels)
# predicted_labels = clf.predict(test_matrix)
with open(OUT_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Id", "Category"])
    for idx, tag in enumerate(predicted_labels):
 csv_list = []
csv_list.append(idx)
csv_list.append(tag)
writer.writerow(csv_list)
