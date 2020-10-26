from flask import Flask, redirect, url_for, request



app = Flask(__name__)

validJobID = 0

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


@app.rout('/train', methods=['POST'])
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
    print('job_info() called!')
    # get job id from args
    # return job status/progress

# run the server
if __name__ == "__main__":
    app.run()