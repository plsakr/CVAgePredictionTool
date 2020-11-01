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


def train(X_young,y_young, X_old, y_old, k_cross_validation_ratio, testing_size, optimal_k=True, min_range_k=1, max_range_k=100, progressObject=None, jobId=-1, model_name="pretrained_knn_model"):

    #X0_train, X_test, y0_train, y_test = train_test_eq_split(X, y, n_per_class=2500, random_state=123)

    X0_young_train, X0_young_test, y0_young_train, y0_young_test= train_test_split(X_young, y_young, test_size=testing_size, random_state=7)
    X0_old_train, X0_old_test, y0_old_train, y0_old_test= train_test_split(X_old, y_old, test_size=testing_size, random_state=7)
    X0_train = X0_young_train.append(X0_old_train)
    y0_train = y0_young_train.append(y0_old_train)
    
    X_test = X0_young_test.append(X0_old_test)
    y_test = y0_young_test.append(y0_old_test)

    df_train=(pd.concat([X0_train, y0_train], axis=1)).sample(frac=1).reset_index(drop=True)
    X0_train = df_train.drop(['age_new'], axis=1)
    y0_train = df_train['age_new']
    print("Hiiiiii")
    print(np.abs(y0_young_train.shape - y0_old_train.shape) >2)

    if(np.abs(y0_young_train.shape - y0_old_train.shape) >2 ):
        raise ValueError('Error! Biased Training dataset!')
    
    eval_score_list = []

    if optimal_k and min_range_k>0 and max_range_k>min_range_k:
        k_range= range(min_range_k, min(max_range_k, X0_train.shape[0]))
    elif optimal_k:
        k_range=range(1,min(50, X0_train.shape[0]))
    else:
        k_range=[min_range_k]
    

    scores = {}
    scores_list = []

    #finding the optimal nb of neighbors
    for k in tqdm(k_range):
        knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance' ,algorithm='ball_tree', p=1)
        knn.fit(X0_train, y0_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
        if progressObject != None:
            progressObject['jobProgress'][jobId] = min((k_range.index(k)+1.0)/len(k_range), 0.99)
    
    k_optimal = scores_list.index(max(scores_list))+min_range_k
    model = KNeighborsClassifier(n_neighbors= k_optimal, weights = 'distance', algorithm='ball_tree', p=1)
    
    accuracys=[]

    nb_splits=10
    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, y0_train)

    old_train_X = []
    old_train_y= []
    old_eval_x = []
    old_eval_y= []
    young_train_X = []
    young_train_y=[]
    young_eval_x = []
    young_eval_y =[]
    for train_old_index, test_old_index in skf.split(X0_old_train, y0_old_train):
        X_train_old, X_eval_old = pd.DataFrame(X0_old_train).iloc[train_old_index], pd.DataFrame(X0_old_train).iloc[test_old_index]
        y_train_old, y_eval_old = pd.DataFrame(y0_old_train).iloc[train_old_index], pd.DataFrame(y0_old_train).iloc[test_old_index]

        old_train_X.append(X_train_old)
        old_train_y.append(y_train_old)
        old_eval_x.append(X_eval_old)
        old_eval_y.append(y_eval_old)
    for train_young_index, test_young_index in skf.split(X0_young_train, y0_young_train):
        X_train_young, X_eval_young = pd.DataFrame(X0_young_train).iloc[train_young_index], pd.DataFrame(X0_young_train).iloc[test_young_index]
        y_train_young, y_eval_young = pd.DataFrame(y0_young_train).iloc[train_young_index], pd.DataFrame(y0_young_train).iloc[test_young_index]
        young_train_X.append(X_train_young)
        young_train_y.append(y_train_young)
        young_eval_x.append(X_eval_young)
        young_eval_y.append(y_eval_young)

    
    for i in range(nb_splits):
        X_train = pd.DataFrame(old_train_X[i]).append(pd.DataFrame(young_train_X[i]))
        X_eval = pd.DataFrame(old_eval_x[i]).append(pd.DataFrame(young_eval_x[i]))
        y_train = pd.DataFrame(old_train_y[i]).append(pd.DataFrame(young_train_y[i]))
        y_eval = pd.DataFrame(old_eval_y[i]).append(pd.DataFrame(young_eval_y[i]))



        df0_train=(pd.concat([X_train, y_train], axis=1)).sample(frac=1).reset_index(drop=True)
        X_train = df0_train.drop(['age_new'], axis=1)
        y_train = df0_train['age_new'] 

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

def train2(X,y, k_cross_validation_ratio, testing_size, optimal_k=True, min_range_k=1, max_range_k=100, progressObject=None, jobId=-1, model_name="pretrained_knn_model"):
    


    X0_train, X_test, y0_train, y_test= train_test_split(X, y, test_size=0.2, random_state=7)
    
    eval_score_list = []

    if optimal_k and min_range_k>0 and max_range_k>min_range_k:
        k_range= range(min_range_k, min(max_range_k, X0_train.shape[0]))
    elif optimal_k:
        k_range=range(1, min(50, X0_train.shape[0]))
    else:
        k_range=[min_range_k]
    

    scores = {}
    scores_list = []

    #finding the optimal nb of neighbors
    for k in tqdm(k_range):
        knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance', algorithm='ball_tree', p=1)
        knn.fit(X0_train, y0_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
        if progressObject != None:
            progressObject['jobProgress'][jobId] = min((k_range.index(k)+1.0)/len(k_range), 0.99)
    
    k_optimal = scores_list.index(max(scores_list))+min_range_k
    model = KNeighborsClassifier(n_neighbors= k_optimal, weights = 'distance', algorithm='ball_tree', p=1)
    #model.fit(X0_train, y0_train)
    accuracys=[]

    nb_splits=10

    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, y0_train)
    for train_index, test_index in skf.split(X0_train, y0_train):
    
        print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = pd.DataFrame(X0_train).iloc[train_index], pd.DataFrame(X0_train).iloc[test_index]
        y_train, y_eval = pd.DataFrame(y0_train).iloc[train_index], pd.DataFrame(y0_train).iloc[test_index]
    
        model.fit(X_train, y_train)
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

    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    test_score = metrics.accuracy_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)
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
                raise ValueError('Error! No face was detected. Please try again with a clearer picture')
                 
                

        print("Count: empty: {}".format(count))
        # the age groups we decide we call 'young'
        young = ['age_10_14', 'age_15_19','age_20_24']
        # in this column, true means young, false means old
        lbp_df['age_new'] = lbp_df['ageRange'].isin(young)
        lbp_df.to_csv('./lbp.csv')

    df_young = lbp_df[lbp_df.age_new == True]
    df_old = lbp_df[lbp_df.age_new == False]
 
    # Split the dataframe into old and young for balanced training and validation
    X_young_df = df_young.drop(['ageRange','age_new'], axis=1)
    y_young_df = df_young['age_new']
    X_old_df = df_old.drop(['ageRange','age_new'], axis=1)
    y_old_df = df_old['age_new']

    return X_young_df, y_young_df ,X_old_df, y_old_df, count

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
    lbp_df = pd.DataFrame()
    # the parameters of the LBP algo
    # higher = more time required
    sample_points = 16
    radius = 4
    images_crop = []
    count=0
    for i in tqdm(imagePaths):
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

def createTrainDFFromBase64(images, isYoung, progressObject, jobId):
    lbp_df = pd.DataFrame()
    sample_points = 16
    radius = 4
    progressObject['jobProgress'][jobId] = 0.0

    for base64Image in images:
        encoded_data = base64Image.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces, x,y,w,h = locateFace(img)
        if not faces == ():
            crop_img = img[y:y+h, x:x+w]
            lbp = LocalBinaryPatterns(sample_points, radius).describe(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
            row = dict(zip(range(0, len(lbp)), lbp))
            row['age_new'] = isYoung
            lbp_df = lbp_df.append(row, ignore_index=True)

        if progressObject != None:
            progressObject['jobProgress'][jobId] = min((images.index(base64Image)+1.0)/len(images), 0.99)

    lbp_df['age_new'] = lbp_df['age_new'].astype('bool')
    X = lbp_df.drop(['age_new'], axis=1)
    y = lbp_df['age_new']

    return X,y
    

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
        X_young,y_young, X_old, y_old, _ =dataframeCreation(images)
        eval_accuracy, model, X_train, y_train, X_test, y_test, k_optimal = train(X_young,y_young, X_old, y_old, k_cross_validation_ratio=5, testing_size=0.2, optimal_k=True, min_range_k= 1, max_range_k=100)
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
            sharedObject['trainingId'] = job['jobID']
            print("training new model based on our dataset")
            model_name="user_knn_model"
            X, y = getRandomSamples(job['nbrYoung'], job['nbrOld'])

            optimize = job['optimizeK']
            if optimize:
                maxK = job['maxK']
            else:
                maxK = 100

            eval_accuracy, model, X_train, y_train, X_test, y_test, k_optimal = train2(X, y, k_cross_validation_ratio=5, testing_size=job['test_ratio'], optimal_k=job['optimizeK'], min_range_k= job['minK'], max_range_k=maxK,model_name=model_name, progressObject=sharedObject, jobId=job['jobID'])
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

        elif job['trainType'] == 'reset':
            print('Resetting model to pretrained!')
            model_name='pretrained_knn_model'
            sharedObject['model_name'] = model_name
            with open('./config.json', 'r+') as f:
                config = json.load(f)
                f.seek(0)
                config['model_name'] = model_name
                json.dump(config, f)
                f.truncate()
            sharedObject['jobProgress'][job['jobID']] = 1.0
        elif job['trainType'] == 'custom':
            sharedObject['isTraining'] = True
            sharedObject['trainingId'] = job['jobID']
            print('Creating model from custom dataset!')
            model_name="user_knn_model"

            imagesYoung = job['youngPics']
            X_young, y_young = createTrainDFFromBase64(imagesYoung, True, sharedObject, job['jobID'])

            imagesOld = job['oldPics']
            X_old, y_old = createTrainDFFromBase64(imagesOld, False, sharedObject, job['jobID'])

            optimize = job['optimizeK']
            if optimize:
                maxK = job['maxK']
            else:
                maxK = 100
            eval_accuracy, model, X_train, y_train, X_test, y_test, k_optimal = train(X_young,y_young, X_old, y_old, k_cross_validation_ratio=5, testing_size=job['test_ratio'], optimal_k=job['optimizeK'], min_range_k=job['minK'], max_range_k=maxK, model_name=model_name, progressObject=sharedObject, jobId=job['jobID'])
            test_score, conf_rep = test(X_train, y_train,X_test, y_test, modelName=model_name)

            print(conf_rep)
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



def predictFromMLScriptOnly(path):
    images = preprocessingData(path)
    X_young_df, y_young_df ,X_old_df, y_old_df, _ =dataframeCreation(images)
    #model = loadPreTrained(modelName="pretrained_knn_model")
    eval_accuracy, model, X_train, y_train, X_test, y_test, k_optimal = train(X_young_df, y_young_df ,X_old_df, y_old_df, k_cross_validation_ratio=5, testing_size=0.2, optimal_k=True, min_range_k= 1, max_range_k=100)
    
    print(pd.DataFrame(y_train).apply(pd.value_counts))
    test_score, conf_rep = test(X_train, y_train,X_test, y_test, modelName="pretrained_knn_model")
    print("Test Score: {}".format(test_score))
    print(conf_rep)

    model.fit(X_train, y_train)
    #model = loadPreTrained("pretrained_knn_model")
    testPath = '../dataset/male/age_10_14/pic_0014.png'
    testPath2 = "../dataset/male/age_50_54/pic_0028.png"
    patht = [testPath, testPath2]
    X=createInputsFromImagePaths(patht)
    label , pred = predict(X, model)
    print(pred)