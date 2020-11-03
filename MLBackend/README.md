# CV Age Prediction backend

This is the backend webserver of the application. It is based on the Flask python library. It also makes use of a machine learning subprocess
(using the `multiprocessing` python library), in order to make requests and receive results to and from the models that are trained and loaded.

## Installing Python Requirements

The backend requires numerous Python libraries in order to run, which are all listed in the `requirements.txt` file that is located in the root
of the repository.

To install the dependencies, we can use the simple command `pip install -r requirements.txt` in a terminal that is navigated to the root folder.
We also recommend using a python virtual environment so that other libraries do not clash with the application. To learn more about this, click
[here](https://docs.python.org/3/library/venv.html)

## Running the backend

To run the backend, navigate to this folder and run `python backendserver.py`. This will start the backend, and create the machine learning subprocess. If the config file (config.json) does not exist, the subprocess will train the initial model, and as such will take a few moments to finish.
