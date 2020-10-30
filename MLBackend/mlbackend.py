import numpy as np 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import glob
import pickle
import base64

from skimage import feature
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score


# Read the input image
def locateFace(img):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    
    for (x, y, w, h) in faces:
        cv2.rectangle(np.copy(img), (x, y), (x+w, y+h), (255, 0, 0), 2)
    if faces == ():
        x=-1
        y=-1
        w=-1
        h=-1
    return faces, x,y,w,h

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist = hist / (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist


def train(X, y, k_cross_validation_ratio, testing_size, optimal_k=True, min_range_k=1, max_range_k=100,model_name="pretrained_knn_model", progressObject=None, jobId=-1):

    X0_train, X_test, y0_train, y_test = train_test_split(X,y,test_size=testing_size, random_state=7)
    
    eval_score_list = []

    if optimal_k and min_range_k>0 and max_range_k>min_range_k:
        k_range= range(min_range_k, max_range_k)
    elif optimal_k:
        k_range=range(1,50)
    else:
        k_range=[min_range_k]
    

    scores = {}
    scores_list = []

    #finding the optimal nb of neighbors
    for k in tqdm(k_range):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X0_train, y0_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
        if progressObject != None:
            progressObject['jobProgress'][jobId] = min((k_range.index(k)+1.0)/len(k_range), 0.99)
    
    k_optimal = scores_list.index(max(scores_list))+min_range_k
    model = KNeighborsClassifier(n_neighbors= k_optimal)

    accuracys=[]

    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, y0_train)
    for train_index, test_index in skf.split(X0_train, y0_train):
    
        # print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = pd.DataFrame(X0_train).iloc[train_index], pd.DataFrame(X0_train).iloc[test_index]
        y_train, y_eval = pd.DataFrame(y0_train).iloc[train_index], pd.DataFrame(y0_train).iloc[test_index]
    
        model.fit(X_train, y_train.values.ravel())
        predictions = model.predict(X_eval)
        score = accuracy_score(predictions, y_eval)
        accuracys.append(score)
        print("Validation batch score: {}".format(score))

    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model.fit(X0_train, y0_train)
    pickle.dump(model, open(model_name, 'wb'))

    return eval_accuracy, model, X0_train, y0_train, X_test, y_test, k_optimal


def loadPreTrained(modelName):
    if os.path.exists(modelName):
        return pickle.load(open(modelName, 'rb' ))
    else:
        strExcep = "Error! Pretrained model does not exist! "
        return strExcep


def test(X_train, y_train, X_test, y_test,modelName):
    model = loadPreTrained(modelName)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Predictions shape: {}".format(y_pred.shape))
    # print("Y_test shape: {}".format(y_test))
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    test_score = metrics.accuracy_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)
    # print(type(classification_rep))

    return  test_score, classification_rep


def preprocessingData(file_path):
    images = []
    for x, y, z in os.walk(file_path):
        for name in tqdm(z):
            images.append(os.path.join(x, name).replace('\\','/'))
    return images 

def dataframeCreation(images):
    if os.path.exists('./lbp.csv'):
        lbp_df = pd.read_csv('./lbp.csv', index_col=0)
        count = 0
    else:
        lbp_df = pd.DataFrame()
        # the parameters of the LBP algo
        # higher = more time required
        sample_points = 16
        radius = 4
        images_crop = []
        count=0
        for i in tqdm(images):
            if ".DS_Store" in i:
                continue  
            img = cv2.imread(i)
            faces, x,y,w,h = locateFace(img)
            if not faces == ():
                crop_img = img[y:y+h, x:x+w]
                lbp = LocalBinaryPatterns(sample_points, radius).describe(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
                row = dict(zip(range(0, len(lbp)), lbp))
                row['ageRange'] = i.split('/')[3] ## TODO: change 3 to the index in the path where the age range is located
                lbp_df = lbp_df.append(row, ignore_index=True)

            else:
                count= count+1
                os.remove(i)
                raise ValueError('Error! No face was detected. Please try again with a clearer picture')
                #to remove the remove method    
                

        print("Count: empty: {}".format(count))
        # the age groups we decide we call 'young'
        young = ['age_10_14', 'age_15_19','age_20_24']
        # in this column, true means young, false means old
        lbp_df['age_new'] = lbp_df['ageRange'].isin(young)
        lbp_df.to_csv('./lbp.csv')

    
    # randomize the df so that old and young are mixed
    random_df = lbp_df.sample(frac=1).reset_index(drop=True)
    random_df.head()
    X = random_df.drop(['ageRange','age_new'], axis=1)
    y = random_df['age_new']
    # print(lbp_df)

    return X,y, count

def getRandomSamples(nbrYoung, nbrOld):
    if os.path.exists('./lbp.csv'):
        full_df = pd.read_csv('./lbp.csv', index_col=0)

        youngs = full_df.loc[full_df['age_new'] == True]
        olds = full_df.loc[full_df['age_new'] == False]

        random_youngs = youngs.sample(n=nbrYoung).reset_index(drop=True)
        random_olds = olds.sample(n=nbrOld).reset_index(drop=True)

        full_random = random_youngs.append(random_olds)

        X = full_random.drop(['ageRange','age_new'], axis=1)
        y = full_random['age_new']
        return X,y


def createInputsFromImagePaths(imagePaths):
#images = preprocessingData(imagePath)
    #image = cv2.imread(imagePath)
    lbp_df = pd.DataFrame()
    # the parameters of the LBP algo
    # higher = more time required
    sample_points = 16
    radius = 4
    images_crop = []
    count=0
    for i in tqdm(imagePath):
        if ".DS_Store" in i:
            continue  
        img = cv2.imread(i)
        faces, x,y,w,h = locateFace(img)
        if not faces == ():
            crop_img = img[y:y+h, x:x+w]
            lbp = LocalBinaryPatterns(sample_points, radius).describe(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
            row = dict(zip(range(0, len(lbp)), lbp))
            lbp_df = lbp_df.append(row, ignore_index=True)
    
        else:
            count= count+1
            print("Error! No face was detected. Please try again with a clearer picture")
            #to remove the remove method    
            os.remove(i)
       
    print("Count: empty: {}".format(count))
    return lbp_df

def createInputFromBase64(base64Image):
    
    encoded_data = base64Image.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    lbp_df = pd.DataFrame()
    # the parameters of the LBP algo
    # higher = more time required
    sample_points = 16
    radius = 4
    images_crop = []
    faces, x,y,w,h = locateFace(img)
    if not faces == ():
        crop_img = img[y:y+h, x:x+w]
        lbp = LocalBinaryPatterns(sample_points, radius).describe(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
        row = dict(zip(range(0, len(lbp)), lbp))
        lbp_df = lbp_df.append(row, ignore_index=True)

    else:
        print("Error! No face was detected. Please try again with a clearer picture")
        
    return lbp_df

def predict(X, model):
    predictions = model.predict(X)
    labels = list()
    lbl = []
    for predict in predictions:
        if predict:
            label = "Young"
        else:
            label = "Old"
        lbl.append(label)
    
    labels.append(lbl)
        
    return labels, predictions

def predictFromPath(imagePath, model):

    X = createInputsFromImagePaths(imagePath)
    # print(lbp_df)
    return predict(X, model)


def getCurrentModel():
    with open('./config.json') as f:
        sharedObject = json.load(f) 
    return loadPreTrained(sharedObject['model_name'])



def predictFromBase64(base64Image):
    model = getCurrentModel()
    X = createInputFromBase64(base64Image)

    return predict(X, model)



def initializeML(initial_dataset_path = './dataset'):
    sharedObject = {}
    model_name = 'pretrained_knn_model'

    if os.path.exists('./config.json') and os.path.exists(model_name):
        print('Found previously saved config!')
        with open('./config.json') as f:
            sharedObject = json.load(f)
    else:
        # load pre-packages model, and calculate its initial scores based on the dataset
        print('No config found, assuming first launch!')
        
        images = preprocessingData(initial_dataset_path)
        X,y, _ =dataframeCreation(images)
        eval_accuracy, model, X_train, y_train, X_test, y_test, k_optimal = train(X, y, k_cross_validation_ratio=5, testing_size=0.2, optimal_k=True, min_range_k= 1, max_range_k=100)
        test_score, conf_rep = test(X_train, y_train,X_test, y_test, modelName=model_name)

        # setup the initial config
        sharedObject['model_name'] = model_name
        sharedObject['p_model_scores'] = {'Young': conf_rep['True'], 'Old': conf_rep['False'], 'acc': conf_rep['accuracy']}
        sharedObject['p_model_params'] = {'K': k_optimal, 'train_nbr': X_train.shape[0], 'test_nbr': X_test.shape[0]}

        with open('./config.json', 'w') as f:
            json.dump(sharedObject, f)

    return sharedObject

def performJob(job, sharedObject):
    jobType = job['type']
    print(type(sharedObject))
    if jobType == 'PREDICT':
        image = job['image']
        labels, _ = predictFromBase64(image)
        print('prediction finished', labels)
        sharedObject['jobProgress'][job['jobID']] = 1.0
        sharedObject['jobResults'][job['jobID']] = {'label': labels[0]}
    if jobType == 'TRAIN':
        if job['trainType'] == 'params':
            sharedObject['isTraining'] = True
            print("training new model based on our dataset")
            model_name="user_knn_model"
            X, y = getRandomSamples(job['nbrYoung'], job['nbrOld'])

            eval_accuracy, model, X_train, y_train, X_test, y_test, k_optimal = train(X, y, k_cross_validation_ratio=5, testing_size=job['test_ratio'], optimal_k=job['optimizeK'], min_range_k= job['minK'], max_range_k=job['maxK'],model_name=model_name, progressObject=sharedObject, jobId=job['jobID'])
            test_score, conf_rep = test(X_train, y_train,X_test, y_test, modelName=model_name)

            sharedObject['model_name'] = model_name
            sharedObject['u_model_scores'] = {'Young': conf_rep['True'], 'Old': conf_rep['False'], 'acc': conf_rep['accuracy']}
            sharedObject['u_model_params'] = {'K': k_optimal, 'train_nbr': X_train.shape[0], 'test_nbr': X_test.shape[0]}

            with open('./config.json', 'r+') as f:
                config = json.load(f)
                f.seek(0)
                config['model_name'] = model_name
                config['u_model_scores'] = {'Young': conf_rep['True'], 'Old': conf_rep['False'], 'acc': conf_rep['accuracy']}
                config['u_model_params'] = {'K': k_optimal, 'train_nbr': X_train.shape[0], 'test_nbr': X_test.shape[0]}
                json.dump(config, f)
                f.truncate()
            sharedObject['jobProgress'][job['jobID']] = 1.0
            sharedObject['isTraining'] = False
            print('finishedTraining!')


    
    

    







    




    