# Computer Vision Age Detector

This repository contains all the code needed for the development, and running an image age detector, based on machine learning. It uses a KNN model based on the Local Binary Patterns of the images of a face to determine whether a subject is old or young.

The project uses an ML Backend based on the `scikit-learn` python library, and utilizes a React Frontend as a UI, along with a Flask backend server to listen for and respond to requests.

## Running The Code

To run the code, one must start the backend server (file: `MLBackend/backendserver.py`) and the frontend server (node project in folder names frontend). More detailed instructions are available for each of the frontend and the backend in their respective README files (`frontend/README.md` and `MLBackend/README.md`)

## Credits

This project was written by Peter Louis Sakr, Melissa Chehade and Samar Saleme as part of the Intelligent Engineering Algorithms course project at LAU.

## License

MIT License

Copyright (c) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
