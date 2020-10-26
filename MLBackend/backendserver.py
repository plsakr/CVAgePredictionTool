from flask import Flask, redirect, url_for, request, Response
import multiprocessing
from queue import SimpleQueue




app = Flask(__name__)

validJobID = 0
currentJobs = []

if __name__ == '__main__':
    print('__name__ is', __name__, 'im supposed to be __main__')
    manager = multiprocessing.Manager()
    jobQueue = manager.Queue() # create a thread-safe queue for background ML tasks # create a thread-safe queue for background ML tasks
    sharedObject = manager.dict()
else:
    print('__name__ is', __name__)
    # import trainlbp_knn # import ml backend in the background! => YOU CANNOT USE ANY OF ITS METHODS/OBJECTS EXCEPT IN background_worker()

def get_next_job_id():
    global validJobID, sharedObject
    currentId = validJobID
    validJobID = validJobID + 1

    if 'jobProgress' in sharedObject:
        sharedObject['jobProgress'][currentId] = 0.0
    else:
        sharedObject['jobProgress'] = {currentId: 0.0}
    print(sharedObject)
    return currentId


@app.route('/info', methods=['GET'])
def get_model_info():
    print('get_model_info() called!')
    return {'training': 200, 'testing': 50, 'tPrecision': 0.9, 'tRecall': 0.89, 'fPrecision': 0.9, 'fRecall': 0.89}


@app.route('/predict', methods=['POST'])
def predict_age():
    print('predict_age() called!')
    # TODO: get needed arguments from request
    res = {'faceDetected': True}

    # TODO: add actual prediction
    res['prediction'] = 'old'
    return res


@app.route('/train', methods=['POST'])
def train_model():
    # get actual train type
    # if reset
    # reset
    return {'jobDone': True}
    # if params
    # get params and create job with id
    job_id = get_next_job_id()
    return {'jobDone': False, 'jobID': job_id}
    #if dataset
    # get params and create job with id
    job_id = get_next_job_id()
    return {'jobDone': False, 'jobID': job_id}


@app.route('/jobinfo', methods=['GET'])
def get_job_info():
    global jobQueue, sharedObject
    print('job_info() called!')
    jobId = request.args.get('jobId', default=None)
    print('jobId', jobId)
    if jobId == None:
        print('A bad request was caught!')
        return Response(status=400)
    
    if 'jobProgress' in sharedObject and jobId in sharedObject['jobProgress']:
        progress = sharedObject['jobProgress'][jobId]
        return {'jobDone': progress == 1.0, 'jobProgress': progress}
    else:
        print('received an invalid job id!')
        return Response({'jobId': jobId}, status=400)


def background_worker(jobQueue, sharedObject):
    # initialize ml backend
    print("background worker started!")
    while True:
        print('looping!')
        while jobQueue.empty():
            pass
        nextJob = jobQueue.get(block=True) # wait here until a job is available to complete
        print('Found a job! Executing now')
        

        

# run the server
if __name__ == '__main__':
    print('starting background jobs')
    myJob = multiprocessing.Process(target=background_worker, args=(jobQueue,sharedObject))
    myJob.start()
    app.run()