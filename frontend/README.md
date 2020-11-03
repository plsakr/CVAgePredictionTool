# CV Age Prediction Frontend

The frontend of the app was created using React. It shows the information of the currently selected model (scores and training information),
as well as allow the user to predict the label of an image (young or old), and finally train a new model, either by selecting a number of random
samples from each of the young and old datasets, or creating a custom dataset from user uploaded images.

## Installing Dependencies

The app requires having NodeJs installed on your system. In addition, we used some third-party libraries for some components of the app.
To download theses dependencies, navigate to this folder using the system terminal and run `npm install`. This should download and allow the use
of all of the required dependencies.

## Running the Frontend

To run the frontend, after having downloaded the requirements, just run `npm start` in this folder. The code should compile, and automatically
open a browser to the correct link (http://localhost:3000). Note that the backend must be running first, as the first render of the app depends
on an initial response from the backend to display updated model information.
