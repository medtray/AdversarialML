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
allPoisons = []
allAdv = []
alldiffs = []
directoryForPoisons = './poisonImages/'
if not os.path.exists(directoryForPoisons):
    os.makedirs(directoryForPoisons)

directoryForadv = './advImages/'
if not os.path.exists(directoryForadv):
    os.makedirs(directoryForadv)

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

        #img, diff = do_optimization(targetImg, baseImg, MaxIter=1500, coeffSimInp=0.2, saveInterim=False, imageID=i,
                                    #objThreshold=2.9)
        img,img2, diff = find_poison_adversarial(targetImg, baseImg, MaxIter=1500, coeffSimInp=0.2,coeffTar=0.1, saveInterim=False, imageID=i,
                                    objThreshold=1)
        print('built poison for target %d with diff: %.5f' % (i, diff))
        counter += 1
    # save the image to file and keep statistics
    allPoisons.append(img)
    allAdv.append(img2)
    alldiffs.append(diff)
    name = "%d_%.5f" % (i, diff)
    misc.imsave(directoryForPoisons + name + '.jpeg', img)
    misc.imsave(directoryForadv + name + '.jpeg', img2)

allPoisons = np.array(allPoisons)
allAdv = np.array(allAdv)
alldiffs = np.array(alldiffs)
np.save('all_poisons.npy', allPoisons)
np.save('all_adv.npy', allAdv)
np.save('alldiffs.npy', alldiffs)
print 'done'