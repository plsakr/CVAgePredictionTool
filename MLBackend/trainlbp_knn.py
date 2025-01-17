import numpy as np 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
from skimage import feature
from tqdm import tqdm
import glob
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut #for P-cross validation
from sklearn.metrics import classification_report, accuracy_score

# Load the cascade
face_cascade = cv2.CascadeClassifier('./lbptest/haarcascade_frontalface_alt2.xml')
profile_face_cascade= cv2.CascadeClassifier('./lbptest/haarcascade_profileface.xml')
eyes_cascade = cv2.CascadeClassifier('./lbptest/frontalEyes.xml')


def imageResizing(img, scalingFactor):
    #resizing the img 
    scaling_factor = scalingFactor
    
    assert not isinstance(img,type(None)), 'image not found'
    width = int(img.shape[1] * scaling_factor/100)
    height = int(img.shape[0] * scaling_factor/100)
    newDimensions = (width,height)

    resizedImg = cv2.resize(img, newDimensions, interpolation = cv2.INTER_AREA)
    return resizedImg


# Read the input image
def locateFace(img):

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    
    for (x, y, w, h) in faces:
        cv2.rectangle(np.copy(img), (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    #cv2.imshow('img', img)
    #cv2.waitKey()
    if faces == ():
        x=-1
        y=-1
        w=-1
        h=-1
    return faces, x,y,w,h

def locateProfile(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = profile_face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(np.copy(img), (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    #cv2.imshow('img', img)
    #cv2.waitKey()
    if faces == ():
        x=-1
        y=-1
        w=-1
        h=-1
    return faces, x, y, w, h

def locateEyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    eyes = profile_face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces

    for (x, y, w, h) in eyes:
        cv2.rectangle(np.copy(img), (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    #cv2.imshow('img', img)
    #cv2.waitKey()
    if eyes == ():
        x=-1
        y=-1
        w=-1
        h=-1
    return eyes, x, y, w,h



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
    #Scaler is needed to scale all the inputs to a similar range
    scaler = StandardScaler()
    scaler = scaler.fit(X0_train)
    X0_train = scaler.transform(X0_train)
    X_test = scaler.transform(X_test)
    #X_train, X_eval, y_train, y_eval = train_test_split(X0_train, y0_train, test_size= 100/k_cross_validation_ratio, random_state=7)
    

    #finding the range for the optimal value of k either within the specified range (user input) 
    # or by our default range
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



    eval_score_list = []
    #Evaluation using cross validation: lpo: leave p out
    from sklearn.model_selection import StratifiedKFold
    lpo = LeavePOut(p=1)
    accuracys=[]

    skf = StratifiedKFold(n_splits=10, random_state=None)
    skf.get_n_splits(X0_train, y0_train)
    for train_index, test_index in skf.split(X0_train, y0_train):
    
        # print("TRAIN:", train_index, "Validation:", test_index)
        X_train, X_eval = pd.DataFrame(X0_train).iloc[train_index], pd.DataFrame(X0_train).iloc[test_index]
        y_train, y_eval = pd.DataFrame(y0_train).iloc[train_index], pd.DataFrame(y0_train).iloc[test_index]
    
        model.fit(X0_train, y0_train)
        predictions = model.predict(X_eval)
        score = accuracy_score(predictions, y_eval)
        accuracys.append(score)
        #scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        #eval_score_list.append(scores.mean())

    #eval_accuracy = np.mean(eval_score_list)
    eval_accuracy = np.mean(accuracys)

    #save the pretrained model:
    model_name='pretrained_knn_model'
    pickle.dump(model, open(model_name, 'wb'))

    return eval_accuracy, model, X0_train, y0_train, X_test, y_test


def test(X_train, y_train, X_test, y_test,pretrain_model=False):
    model_name='pretrained_knn_model'
    if pretrain_model:
        model = pickle.load(open(model_name, 'rb' ))
        
    else:
        eval_score, model, X_train, y_train, X_test, y_test = train(X_test, y_test, pretrained_model=False)
        print("Evaluation score: {}".format(eval_score))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Predictions shape: {}".format(y_pred.shape))
    print("Y_test shape: {}".format(y_test))
    classification_rep = classification_report(y_test, y_pred)
    test_score = metrics.accuracy_score(y_test, y_pred)

    return test_score, classification_rep

# if __name__ == '__main__':
path = './dataset/male' ## TODO: Change this to the path of your dataset. (The code will look through every subfolder for images)

images = []
for x, y, z in os.walk(path):
    for name in tqdm(z):
        images.append(os.path.join(x, name).replace('\\','/')) 
        

lbp_df = pd.DataFrame()
# the parameters of the LBP algo
# higher = more time required
sample_points = 16
radius = 4

# this code takes a while
images_crop = []
# this code takes a while
count_empty=0

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
        ## for example, mine was ../dataset/female/age_10_14/imagename => split by / => index 3
        lbp_df = lbp_df.append(row, ignore_index=True)
    
    else:
        # profile_face, x,y,w,h = locateProfile(img)
        # if not profile_face == ():
        #     crop_img = img[y:y+h, x:x+w]
        #     lbp = LocalBinaryPatterns(sample_points, radius).describe(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
        #     row = dict(zip(range(0, len(lbp)), lbp))
        #     row['ageRange'] = i.split('/')[3] ## TODO: change 3 to the index in the path where the age range is located
        #     ## for example, mine was ../dataset/female/age_10_14/imagename => split by / => index 3
        #     lbp_df = lbp_df.append(row, ignore_index=True)
        
        # else:
        #     eyes_frontal, x,y,w,h = locateEyes(img)
        #     if not eyes_frontal == ():
        #         crop_img = img[y:y+h, x:x+w]
        #         lbp = LocalBinaryPatterns(sample_points, radius).describe(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
        #         row = dict(zip(range(0, len(lbp)), lbp))
        #         row['ageRange'] = i.split('/')[3] ## TODO: change 3 to the index in the path where the age range is located
        #         ## for example, mine was ../dataset/female/age_10_14/imagename => split by / => index 3
        #         lbp_df = lbp_df.append(row, ignore_index=True)

        #     #neither a frontal view nor a profile view /eyes of the face
        #     else:
        count_empty = count_empty+1

print('Number of not found faces:', count_empty)

# TODO: DELETE UNECESSARY LINES
# number of null values in our df. Should always be 0
# lbp_df[2].isna().sum()

# corrM = lbp_df.corr()
# print(corrM)

# the age groups we decide we call 'young'
young = ['age_10_14',
'age_15_19',
'age_20_24']

# in this column, true means young, false means old
lbp_df['age_new'] = lbp_df['ageRange'].isin(young)

# lbp_df.head()

# randomize the df so that old and young are mixed
random_df = lbp_df.sample(frac=1).reset_index(drop=True)
# random_df.head()


X = random_df.drop(['ageRange','age_new'], axis=1)
y = random_df['age_new']

eval_accuracy, model, X_train, y_train, X_test, y_test = train(X, y, k_cross_validation_ratio=5, testing_size=0.2, optimal_k=True, min_range_k= 1, max_range_k=100)
test_score, conf_rep = test(X_train, y_train,X_test, y_test, pretrain_model=True)
print("Evaluation Score: {}".format(eval_accuracy))
print("Test Score: {}".format(test_score))
print(conf_rep)





    




    