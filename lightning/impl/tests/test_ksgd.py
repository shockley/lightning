from numpy import *
from scipy.sparse import *
import warnings,sklearn.datasets as datasets
from lightning.impl.karma_sgd import KSGDClassifier
from lightning.classification import *

import random

from sklearn.cross_validation import train_test_split

from sklearn.datasets.samples_generator import make_regression

from lightning.impl.datasets.samples_generator import make_classification
from lightning.impl.datasets.samples_generator import make_nn_regression


from sklearn.naive_bayes import MultinomialNB
import sys
sys.path.append('../../../../../synthetic_yahoo_movie/sibm_auto/code/')
import utility as util

MODELS = ["sdca",#"karma",
"nb","adagrad"]

#MODELS = ["sdca","linearsvc","adagrad","sgd"]
lossfunc = "hinge"

# MODELS = ["sag","sgd","adagrad"]
# loss = "log"
sparse = 0.0

def main():
	#f = '../../../../../book_crossing/global_data/user_book_explicit.sparse'
	#f = '../../../../../nf_prize_download/global_data/netflix.sparse'
	f = '../../../../../../rcv1/rcv1_train.binary'
	#f = '../../../../../movie_lens/global_data/ml_1k/ml100k.sparse'
	(x,y) = readfile(f)
	y = array(y, dtype = float64)
	y[y>0.0] = 1.0
	y[y<=0.0] = -1.0
	
	for m in MODELS:
		test(x,y,m)

def readfile(f):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    (x, y) = datasets.load_svmlight_file(f, zero_based=False)
    x = util.sparsify(x, sparse)
  print "data loaded"
  return (x,y)

def test_text():
	(x_tr,y_tr) = readfile(f)
	test(x,y,model)

def test(x,y,model):
	
	rs = int(100 * random.random())
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=rs)

	epoch = 1e2
	large_epoch = 1e3
	tol = 1e-4
	# we downplay regularization / focus on penalty
	regularization = 1e-4
	penalty = 1/regularization
	if model == "karma":
		clf = KSGDClassifier(random_state=rs, loss=lossfunc, learning_rate="pegasos", max_iter=1e1, alpha=1.0, gamma=2.0, shuffle=True, verbose=True)
	elif model == "sdca":
		#regularization = 1.0
		#regularization /= x_train.shape[0]
		clf = SDCAClassifier(alpha = regularization, loss=lossfunc, max_iter=epoch, tol =tol, verbose = True)
	elif model == "sgd":
		clf = SGDClassifier(random_state=rs, loss=lossfunc, learning_rate="pegasos", max_iter=epoch, shuffle=True)
	elif model == "nb":
		clf = MultinomialNB()
	elif model == "linearsvc":
		clf = LinearSVC(C=penalty * len(y_train), loss=lossfunc, tol = tol, max_iter=epoch, verbose = False)
	elif model == "sag":
		clf = SAGClassifier(eta=1.0, alpha=regularization, loss=lossfunc, gamma=1.0, max_iter=epoch, n_inner=1.0, tol=tol, random_state=rs, verbose=0)
	elif model == "adagrad":
		clf = AdaGradClassifier(eta=1.0, alpha=regularization, loss=lossfunc, gamma=1.0, n_iter=epoch, random_state=None)
	clf.fit(x_train, y_train)
	if model != "nb" and model != "karma":
		loss = clf.avg_loss(x_train, y_train)
		print model,"avg train loss", loss
		testloss = clf.avg_loss(x_test, y_test)
		print model,"avg test loss", testloss
		#return
	error = 1.0 - clf.score(x_test, y_test)
	print model,"test error", error



main()