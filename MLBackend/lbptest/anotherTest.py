import numpy as np 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
import pickle

from skimage import feature
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score


path = '../dataset/male' ## TODO: Change this to the path of your dataset. 

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


def train(X, y, k_cross_validation_ratio, testing_size, optimal_k=True, min_range_k=0, max_range_k=0 ):

    X0_train, X_test, y0_train, y_test = train_test_split(X,y,test_size=testing_size, random_state=7)
    
    eval_score_list = []
    if optimal_k and min_range_k>0 and max_range_k>min_range_k:
        k_range= range(min_range_k, max_range_k)
    else:
        k_range=range(1,50)
    

    scores = {}
    scores_list = []

    #finding the optimal nb of neighbors
    for k in tqdm(k_range):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X0_train, y0_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
    
    k_optimal = scores_list.index(max(scores_list))
    model = KNeighborsClassifier(n_neighbors= k_optimal)

    accuracys=[]

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
    model_name='pretrained_knn_model'
    pickle.dump(model, open(model_name, 'wb'))

    return eval_accuracy, model, X0_train, y0_train, X_test, y_test


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
    print("Predictions shape: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(y_test))
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    test_score = metrics.accuracy_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)
    print(type(classification_rep))

    return  test_score, classification_rep


def preprocessingData(file_path=path):
    images = []
    for x, y, z in os.walk(path):
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
                print("Error! No face was detected. Please try again with a clearer picture")
                #to remove the remove method    
                os.remove(i)

        print("Count: empty: {}".format(count))
        # number of null values in our df. Should always be 0
        lbp_df[2].isna().sum()
        corrM = lbp_df.corr()
        #print(corrM)
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
    print(lbp_df)

    return X,y, count


def predict(imagePath, model):

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


    print(lbp_df)
    
    predictions = model.predict(lbp_df)
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



images = preprocessingData(path)
X,y, _ =dataframeCreation(images)
# model = loadPreTrained(modelName="pretrained_knn_model")
eval_accuracy, model, X_train, y_train, X_test, y_test = train(X, y, k_cross_validation_ratio=5, testing_size=0.2, optimal_k=True, min_range_k= 1, max_range_k=100)
X0_train, X_test, y0_train, y_test = train_test_split(X,y,test_size=0.2, random_state=7)
test_score, conf_rep = test(X0_train, y0_train,X_test, y_test, modelName="pretrained_knn_model")
#print("Evaluation Score: {}".format(eval_accuracy))
print("Test Score: {}".format(test_score))
print(conf_rep)

# model.fit(X0_train, y0_train)
#model = loadPreTrained("pretrained_knn_model")
testPath = '../dataset/female/age_15_19/01445.jpg'
testPath2 = '../dataset/female/age_60_94/16883.jpg'
patht = [testPath, testPath2]
label , pred=predict(patht, model)
print(pred)
#print("Test label young: {}".format(label))






    




    