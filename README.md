                                                      
                                               # OBJECT DETECTION 
   # Object Detection
          
 
Abstract
Real-time object detection and tracking is a vast, vibrant yet inconclusive and complex area of computer vision. Due to its increased utilization in surveillance, tracking system used in security and many others applications have propelled researchers to continuously devise more efficient and competitive algorithms. The latest research on this area has been making great progress in many directions.  In the current manuscript, we give an overview of the necessity of object detection in today’s computing systems, outline the current main research directions, discuss on how our API algorithm works, and discuss open problems and possible future directions.
Key words: - Object Detection, Computer Vision, Tracking Systems, API algorithm.


I. INTRODUCTION

A.What is object detection?

Given an image or a video stream, an object detection model can identify which of a known set of objects might be present and provide information about their positions within the image.

An object detection model is trained to detect the presence and location of multiple classes of objects. For example, a model might be trained with images that contain various pieces of fruit, along with a label that specifies the class of fruit they represent (e.g. an apple, a banana, or a strawberry), and data specifying where each object appears in the image.

When we subsequently provide an image to the model, it will output a list of the objects it detects, the location of a bounding box that contains each object, and a score that indicates the confidence that detection was correct.

B. Confidence score

To interpret these results, we can look at the score and the location for each detected object. The score is a number between 0 and 1 that indicates confidence that the object was genuinely detected. The closer the number is to 1, the more confident the model is.
Depending on your application, you can decide a cut-off threshold below which you will discard detection results. For our example, we might decide a sensible cut-off is a score of 0.5 (meaning a 50% probability that the detection is valid). In that case, we would ignore those objects whose scores are below 0.5.
The cut-off you use should be based on whether you are more comfortable with false positives (objects that are wrongly identified, or areas of the image that are erroneously identified as objects when they are not), or false negatives (genuine objects that are missed because their confidence was low).
For example, in the following image, a pear (which is not an object that the model was trained to detect) was misidentified as a "person". This is an example of a false positive that could be ignored by selecting an appropriate cut-off. In this case, a cut-off of 0.6 (or 60%) would comfortably exclude the false positive.

Fig 1.Still Image Object Detection

![](Images/1_KkAZPEPkBNQYTXSFcXwEzA.png)

Fig 2.Live Webcam Object Detection


![](Images/Objects2.jpg)


C. Location
For each detected object, the model will return an array of four numbers representing a bounding rectangle that surrounds its position. For the starter model we provide, the numbers are ordered as follows:
[	top,	left,	bottom,	right	]

The top value represents the distance of the rectangle’s top edge from the top of the image, in pixels. The left value represents the left edge’s distance from the left of the input image. The other values represent the bottom and right edges in a similar manner.


II. WHY DO WE NEED OBJECT DETECTION?


Object recognition allows robots and AI programs to pick out and identify objects from inputs like video and still camera images. Methods used for object identification include 3D models, component identification, edge detection and analysis of appearances from different angles. The first use case is a smarter retail checkout experience. This is a hot field right now after the announcement of Amazon Go stores.
Stores can be designed so they have smart shelves that track what a customer is picking from them. I did this by building two object detection models — one that tracks hand and captures what the hand has picked. And the second independent model that monitors shelf space.  By using two models you minimise the error from a single approach.Another application of computer vision for retail checkout can be that instead of scanning items one by one at a checkout system , everything is placed together and cameras are able to detect and log everything. Maybe we don’t even need a checkout lane. Shopping carts can be equipped with cameras and you can simply walk out with your cart which can bill you as you step out of the store! Won’t this be cool! I used the API to design a “mini” model with 3 random items and the model could easily detect what was placed and in what quantity. See GIF below. Through various experimentation, I found that the API performs very well even on items that are only partially visible.	

III. A GENERAL FRAMEWORK FOR OBJECT DETECTION

Typically, we follow three steps when building an object detection framework:


1.	First, a deep learning model or algorithm is used to generate a large set of bounding boxes spanning the full image (that is, an object localization component)
2.	Next, visual features are extracted for each of the bounding boxes. They are evaluated and it is determined whether and which objects are present in the boxes based on visual features (i.e. an object classification component)
3.	In the final post-processing step, overlapping boxes are combined into a single bounding box (that is, non-maximum suppression)
That’s it – we are ready with your first object detection framework!

A. Packages to be installed:

  1. pip install protobuf
  2. pip install pillow
  3. pip install lxml
  4. pip install Cython
  5. pip install jupyter
  6. pip install matplotlib
  7. pip install pandas
  8. pip install opencv-python 
  9. pip install tensorflow

#B. What is an API? Why do we need an API?

API stands for Application Programming Interface. An API provides developers a set of common operations so that they don’t have to write code from scratch.
In one sense, APIs are great time savers. They also offer users convenience in many cases. Think about it – Facebook users (including myself!) appreciate the ability to sign into many apps and sites using their Facebook ID. How do you think this works? Using Facebook’s APIs of course!
So in this article, we will look at the TensorFlow API developed for the task of object detection.

C. Packages imported description
Pycocotools - Tools for working with the MSCOCO dataset.

OS – The OS module in python provides functions for interacting with the operating system. OS, comes under Python's standard utility modules. This module provides a portable way of using operating system dependent functionality. Python os system input text to script.

Pathlib – Object oriented filesystem path. This module offers classes representing filesystem paths with semantics appropriate for different operating systems. Path classes are divided between pure path which provide purely computational operations without I/O, and concrete path, which inherit from pure paths but also provide I/O operations.

Numpy - NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays.

Six.moves.urlib – The six. moves module provides those modules under a common name for both Python2 and 3.  imports urllib when run with Python3 and imports a mixture of urllib , urllib2 

Sys - System-specific parameters and functions. The sys module provides information about constants, functions and methods of the Python interpreter.

Tarfile - The tarfile module makes it possible to read and write tar archives, including those using gzip, bz2 and lzma compression.

TensorFlow - TensorFlow is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.

Zipfile - The ZIP file format is a common archive and compression standard. This module provides tools to create, read, write, append, and list a ZIP file.

Defaultdict – Dictionary in Python is an unordered collection of data values that are used to store data values like a map. A Dictionary can be created by placing a sequence of elements within curly {} braces, separated by ‘comma’.

StringIo - This module implements a file-like class, StringIo, that reads and writes a string buffer (also known as memory files). See the description of file objects for operations. It can be initialized to an existing string by passing the string to the constructor.

Pyplot – Pyplot is a collection of command style functions that make matplotlib work like MATLAB. Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc.

Image - Python Imaging Library which provides the python interpreter with image editing capabilities.

Display - When this object is returned by an expression or passed to the display function, it will result in the data being displayed in the frontend. The MIME type of the data should match the subclasses used, so the Png subclass should be used for ‘image/png’ data.

Ops - Operator framework charms are just Python code. The entry point to your charm is a particular Python file. It could be anything that makes sense to your project.
Label_map_util – This is needed since the notebook is stored in the object_detection folder.


D. TensorFlow Object Detection API

The TensorFlow object detection API is the framework for creating a deep learning network that solves object detection problems.
There are already pretrained models in their framework which they refer to as Model Zoo. This includes a collection of pretrained models trained on the COCO dataset, the KITTI dataset, and the Open Images Dataset. These models can be used for inference if we are interested in categories only in this dataset.


IV. RESULTS:

We feed the input image to the Object Detection API to generate a convolutional feature map. From the convolutional feature map, we identify the region of proposals and warp them into squares. And by using an RoI (Region of Interest layer) pooling layer, we reshape them into a fixed size so that it can be fed into a fully connected layer.