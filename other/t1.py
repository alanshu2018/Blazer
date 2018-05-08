
#from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn.linear_model import *
#vct= HashingVectorizer()
#clf= SGDRegressor()

import wordbatch
from wordbatch.models import FTRL
from wordbatch.extractors import WordBag
wb= wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams":2, "hash_ngrams_weights":[0.5, -1.0], "hash_size":2**23, "norm":'l2', "tf":'log', "idf":50.0}))
clf= FTRL(alpha=1.0, beta=1.0, L1=0.00001, L2=1.0, D=2 ** 25, iters=1)

train_texts= ["Cut down a tree with a herring? It can't be done.", "Don't say that word.", "How can we not say the word if you don't tell us what it is?"]
train_labels= [1, 0, 1]
test_texts= ["Wait! I said it! I said it! Ooh! I said it again!"]

values = wb.transform(train_texts)
clf.fit(values, train_labels)
preds= clf.predict(wb.transform(test_texts))
print("values={}".format(values))
print("values={}".format(len(values)))
print("texts={}".format(test_texts))
print("transformed={}".format(wb.transform(test_texts)))
print(preds)
