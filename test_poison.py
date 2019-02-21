import numpy as np
import tensorflow as tf
from utils import *
from os import listdir
import imageio
import matplotlib.pyplot as plt



#load the data from file
directorySaving = './final_images/XY/'
all_datas = ['X_tr_feats', 'X_tst_feats', 'X_tr_inp', 'X_tst_inp', 'Y_tr', 'Y_tst']
X_tr = np.load(directorySaving+all_datas[0]+'.npy')
X_test = np.load(directorySaving+all_datas[1]+'.npy')
X_inp_tr = np.load(directorySaving+all_datas[2]+'.npy')
X_inp_test = np.load(directorySaving+all_datas[3]+'.npy')
Y_tr = np.load(directorySaving+all_datas[4]+'.npy')
Y_test = np.load(directorySaving+all_datas[5]+'.npy')
print('done loading data!')

#imgplot = plt.imshow(X_inp_test[0])
#plt.show()

first_time=1
if first_time:
	classes,probs=train_without_poison(X_tr=X_tr,Y_tr=Y_tr,Y_validation=Y_test,X_validation=X_test, cold=True)

	np.save('classes_normal_training.npy',classes)
	np.save('probs_normal_classes.npy',probs)

Poises = np.load('all_poisons.npy')
Adv = np.load('all_adv.npy')
other_class_prob = []
numNotMiscclassified = 0
poison_corr_class_prob = []
numPoisonCorr = 0
print(Poises.shape)
for targID, thePoison in enumerate(Poises):
	print("******************%d********************"%targID)
	target_class_probs, target_corr_pred, poison_class_probs, poison_corr_pred = train_last_layer_of_inception(advimage=Adv[targID],targetFeatRep=X_test[targID],poisonInpImage=thePoison,poisonClass=1-Y_test[targID],X_tr=X_tr,Y_tr=Y_tr,Y_validation=Y_test,X_validation=X_test, cold=True)
	other_class_prob.append(target_class_probs[0][int(1-Y_test[targID])])
	numNotMiscclassified += 1*target_corr_pred[0]
	poison_corr_class_prob.append(poison_class_probs[0][int(1-Y_test[targID])])
	numPoisonCorr += 1*poison_corr_pred[0]
print('##############################')
print('out of %d poisons, %d got correctly classified!'%(len(Poises),numPoisonCorr))
print('out of %d targets, %d got misclassified!'%(len(Y_test),len(Y_test) - numNotMiscclassified))
other_class_prob = np.array(other_class_prob)
np.save('Wrong_ClassProb_targ.npy',other_class_prob)
poison_corr_class_prob = np.array(poison_corr_class_prob)
np.save('Right_ClassProb_poison.npy',poison_corr_class_prob)