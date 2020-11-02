from flask import Flask, redirect, url_for, request, Response
from flask_cors import CORS
import multiprocessing
from queue import SimpleQueue




app = Flask(__name__)
CORS(app)

validJobID = 0
currentJobs = []
if __name__ == '__main__':
    print('__name__ is', __name__, 'im supposed to be __main__')
    manager = multiprocessing.Manager()
    jobQueue = manager.Queue() # create a thread-safe queue for background ML tasks # create a thread-safe queue for background ML tasks
    sharedObject = manager.dict()
    sharedObject['jobProgress'] = manager.dict()
    sharedObject['jobResults'] = manager.dict()
    sharedObject['isTraining'] = False
    sharedObject['trainingId'] = -1
else:
    print('__name__ is', __name__)

def get_next_job_id():
    global validJobID, sharedObject
    currentId = validJobID
    validJobID = validJobID + 1

    if 'jobProgress' in sharedObject:
        sharedObject['jobProgress'][currentId] = 0.0
        sharedObject['jobResults'][currentId] = manager.dict()
    else:
        sharedObject['jobProgress'] = {currentId: 0.0}
        sharedObject['jobResults'] ={currentId: {}}
    return currentId


@app.route('/info', methods=['GET'])
def get_model_info():
    global sharedObject
    print('get_model_info() called!')
    modelName = sharedObject['model_name']

    if modelName == 'pretrained_knn_model':
        return {'isTraining': sharedObject['isTraining'], 'trainingId': sharedObject['trainingId'], 'model_name': modelName, 'model_scores': sharedObject['p_model_scores'], 'model_params': sharedObject['p_model_params']}
    else:
        return {'isTraining': sharedObject['isTraining'], 'trainingId': sharedObject['trainingId'], 'model_name': modelName, 'model_scores': sharedObject['u_model_scores'], 'model_params': sharedObject['u_model_params']}


@app.route('/predict', methods=['POST'])
def predict_age():
    global jobQueue
    print('predict_age() called!')
    
    jobId = get_next_job_id()
    job = {'type': 'PREDICT','jobID': jobId, 'image': request.json['image']}

    jobQueue.put(job)
    return {'jobDone': False, 'jobId': jobId}


@app.route('/train', methods=['POST'])
def train_model():
    global jobQueue

    print('train_model() called!')
    jobId = get_next_job_id()
    if request.json['isReset']:
        job = {'type': 'TRAIN', 'trainType': 'reset', 'jobID': jobId}
        jobQueue.put(job)
        return {'jobDone': False, 'jobId': jobId}
    elif request.json['isCustom']:
        optimize_k = request.json['optimizeK']
        min_k = request.json['minK']
        if optimize_k:
            max_k = request.json['maxK']
        test_ratio = request.json['testRatio']
        youngStrings = request.json['youngPics']
        oldStrings = request.json['oldPics']

        if optimize_k:
            job = {'type': 'TRAIN', 'jobID': jobId, 'trainType': 'custom', 'optimizeK': optimize_k, 'minK': min_k, 'maxK': max_k,
                'youngPics': youngStrings, 'oldPics': oldStrings, 'test_ratio': test_ratio}
        else:
            job = {'type': 'TRAIN', 'jobID': jobId, 'trainType': 'custom', 'optimizeK': optimize_k, 'minK': min_k,
                'youngPics': youngStrings, 'oldPics': oldStrings, 'test_ratio': test_ratio}
        jobQueue.put(job)
        return {'jobDone': False, 'jobId': jobId}

    else:
        optimize_k = request.json['optimizeK']
        min_k = request.json['minK']
        if optimize_k:
            max_k = request.json['maxK']
        nbr_young = request.json['nbrYoung']
        nbr_old = request.json['nbrOld']
        test_ratio = request.json['testRatio']

        if optimize_k:
            job = {'type': 'TRAIN', 'jobID': jobId, 'trainType': 'params', 'optimizeK': optimize_k, 'minK': min_k, 'maxK': max_k,
                'nbrYoung': nbr_young, 'nbrOld': nbr_old, 'test_ratio': test_ratio}
        else:
            job = {'type': 'TRAIN', 'jobID': jobId, 'trainType': 'params', 'optimizeK': optimize_k, 'minK': min_k,
                'nbrYoung': nbr_young, 'nbrOld': nbr_old, 'test_ratio': test_ratio}

        jobQueue.put(job)
        return {'jobDone': False, 'jobId': jobId}
    
    return {'jobDone': False, 'jobID': -1}


@app.route('/jobinfo', methods=['GET'])
def get_job_info():
    global jobQueue, sharedObject
    print('job_info() called!')
    jobId = int(request.args.get('jobId', default=None))
    print('jobId', jobId)
    if jobId == None:
        print('A bad request was caught!')
        return Response(status=400)
    
    if 'jobProgress' in sharedObject and jobId in sharedObject['jobProgress']:
        progress = sharedObject['jobProgress'].copy()[jobId]
        results = sharedObject['jobResults'][jobId].copy()
        return {'jobDone': progress == 1.0, 'jobProgress': progress, 'jobResults': results}
    else:
        print('received an invalid job id!', jobId)
        return Response({'jobId': jobId}, status=400)


def background_worker(jobQueue, sharedObject):
    # initialize ml backend
    print('Initializing background worker')
    import mlbackend
    sharedObject.update(mlbackend.initializeML())
    print("background worker started!")
    while True:
        print('looping!')
        while jobQueue.empty():
            pass
        nextJob = jobQueue.get(block=True) # wait here until a job is available to complete
        print('Found a job! Executing now')
        try:
            mlbackend.performJob(nextJob, sharedObject) 
        except:
            print("error! Did not detect a face")
             


# run the server
if __name__ == '__main__':
    print('starting background jobs')
    myJob = multiprocessing.Process(target=background_worker, args=(jobQueue,sharedObject))
    myJob.start()
    app.run()