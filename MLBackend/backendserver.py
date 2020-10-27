from flask import Flask, redirect, url_for, request
import multiprocessing
from queue import SimpleQueue



app = Flask(__name__)

validJobID = 0
currentJobs = []
manager = multiprocessing.Manager()
jobQueue = manager.Queue() # create a thread-safe queue for background ML tasks


def get_next_job_id():
    global validJobID
    currentId = validJobID
    validJobID = validJobID + 1
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
    global jobQueue
    print('job_info() called!')
    jobQueue.put('hi')
    return {}
    # get job id from args
    # return job status/progress


def background_worker():
    global jobQueue
    print("background worker started!")
    while True:
        print('looping!')
        while jobQueue.empty():
            pass
        nextJob = jobQueue.get(block=True) # wait here until a job is available to complete
        print('Found a job! Executing now')
        

        

# run the server
if __name__ == "__main__":
    print('starting background jobs')
    myJob = multiprocessing.Process(target=background_worker)
    myJob.start()
    app.run()