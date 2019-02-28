from utils import *
import tensorflow as tf
directory = './final_images/'

first_run=0
if first_run:
    dogDir = './ImageNet_Utils/n02084071/n02084071_urlimages/'
    allDogs = load_images_from_directory('dog', dogDir)
    allDogs = clean_data(X=allDogs)
    #save the images to file for future use
    directory = './final_images/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'dogInput.npy',allDogs)

    #get the feature representations and save them
    dogFeats = get_feat_reps(X=allDogs, class_t='dog')
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'dogFeats.npy',dogFeats)

    fishDir = './ImageNet_Utils/n02512053/n02512053_urlimages/'
    allfish = load_images_from_directory('fish', fishDir)
    allfish = clean_data(X=allfish)
    #save the images to file for future use
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'fishInput.npy',allfish)

    #get the feature representations and save them
    fishFeats = get_feat_reps(X=allfish, class_t='fish')
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'fishFeats.npy',fishFeats)

    X_tr, X_test, X_inp_tr, X_inp_test, Y_tr, Y_test = load_bottleNeckTensor_data(directory='./final_images/',
                                                                              saveEm=True)

threshold = 3.5
#load the training and test data
directorySaving = './final_images/XY/'
all_datas = ['X_tr_feats', 'X_tst_feats', 'X_tr_inp', 'X_tst_inp', 'Y_tr', 'Y_tst']
X_tr = np.load(directorySaving+all_datas[0]+'.npy')
X_test = np.load(directorySaving+all_datas[1]+'.npy')
X_inp_tr = np.load(directorySaving+all_datas[2]+'.npy')
X_inp_test = np.load(directorySaving+all_datas[3]+'.npy')
Y_tr = np.load(directorySaving+all_datas[4]+'.npy')
Y_test = np.load(directorySaving+all_datas[5]+'.npy')
print('done loading data i.e. the train-test split!')

# some intializations before we actually make the poisons
allPoisonsOnly = []
allAdvOnly = []
allPoisonsCombined = []
allAdvCombined = []
distortionPoisonsOnly = []
distortionAdvOnly = []
distortionPoisonsCombined_ = []
distortionAdvCombined = []
alldiffs = []
directoryForPoisonsOnly = './poisonImagesOnly/'
if not os.path.exists(directoryForPoisonsOnly):
    os.makedirs(directoryForPoisonsOnly)

directoryForadvOnly = './advImagesOnly/'
if not os.path.exists(directoryForadvOnly):
    os.makedirs(directoryForadvOnly)

directoryForPoisonsCombined = './poisonImagesCombined/'
if not os.path.exists(directoryForPoisonsCombined):
    os.makedirs(directoryForPoisonsCombined)

directoryForadvCombined = './advImagesCombined/'
if not os.path.exists(directoryForadvCombined):
    os.makedirs(directoryForadvCombined)

#len(X_test)
for i in range(1):
    diff = 100
    maxTriesForOptimizing = 10
    counter = 0
    targetImg = X_inp_test[i]
    usedClosest = False
    while (diff > threshold) and (counter < maxTriesForOptimizing):

        if not usedClosest:
            ind = closest_to_target_from_class(classBase=1 - Y_test[i], targetFeatRep=X_test[i],
                                               allTestFeatReps=X_test, allTestClass=Y_test)
            baseImg = X_inp_test[ind]
            usedClosest = True
        else:
            print('Using random base!')
            classBase = 1 - Y_test[i]
            possible_indices = np.argwhere(Y_test == classBase)[:, 0]
            ind = np.random.randint(len(possible_indices))
            ind = possible_indices[ind]
            baseImg = X_inp_test[ind]

        only_poison, diff_poison,diff_poison_with_base = do_optimization(targetImg, baseImg, MaxIter=1500, coeffSimInp=0.2, saveInterim=False, imageID=i,
                                    objThreshold=3)
        only_adv, diff_adv,diff_adv_with_target = do_optimization(baseImg, targetImg, MaxIter=1500, coeffSimInp=0.2, saveInterim=False,
                                            imageID=i,
                                            objThreshold=3)
        img,img2, diff = find_poison_adversarial(targetImg, baseImg, MaxIter=1500, coeffSimInp=0.2,coeffTar=0.1, saveInterim=False, imageID=i,
                                    objThreshold=1)
        print('built poison for target %d with diff: %.5f' % (i, diff))
        counter += 1
    # save the image to file and keep statistics
    allPoisonsOnly.append(only_poison)
    allAdvOnly.append(only_adv)
    allPoisonsCombined.append(img)
    allAdvCombined.append(img2)
    #alldiffs.append(diff)

    name = "%d_%.5f" % (i, diff_poison)
    misc.imsave(directoryForPoisonsOnly + name + '.jpeg', only_poison)
    name = "%d_%.5f" % (i, diff_adv)
    misc.imsave(directoryForadvOnly + name + '.jpeg', only_adv)
    name = "%d_%.5f" % (i, diff)
    misc.imsave(directoryForPoisonsCombined + name + '.jpeg', img)
    misc.imsave(directoryForadvCombined + name + '.jpeg', img2)

allPoisonsOnly = np.array(allPoisonsOnly)
allAdvOnly = np.array(allAdvOnly)
allPoisonsCombined = np.array(allPoisonsCombined)
allAdvCombined = np.array(allAdvCombined)
#alldiffs = np.array(alldiffs)

np.save('all_poisons_only.npy', allPoisonsOnly)
np.save('all_adv_only.npy', allAdvOnly)
np.save('all_poisons_combined.npy', allPoisonsCombined)
np.save('all_adv_combined.npy', allAdvCombined)

#np.save('alldiffs.npy', alldiffs)
print 'done'