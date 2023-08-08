## Table of Contents
- [Problem Statement](#problem-statement)
- [Data Flow](#data-flow)
- [Datasets for Computer Vision and Natural Language Generation](#datasets-for-computer-vision-and-natural-language-generation)
  - [Common Gen](#common-gen)
  - [Common Objects in Context 2017](#common-objects-in-context-2017)
  - [Stanford 40](#stanford-40)
  - [SUN 397](#sun-397)
- [Computer Vision](#computer-vision)
  - [Object Recognition](#object-recognition)
  - [Scene Recognition](#scene-recognition)
  - [Location](#location)
  - [Human Actions](#human-actions)
  - [OCR](#ocr)
- [Natural Language Generation](#natural-language-generation)
- [Streamlit Module](#streamlit-module)
  - [Setup](#setup)
  - [Setup Requirements](setup-requirments)
  - [Beginning with Streamlit Module](#beginning-with-streamlit-module)
  - [Usage](#usage)
  - [Limitations](#limitations)
  - [Development](#development)
   
## Problem Statement
Artificial intelligence (AI) advancements present opportunities to develop improved assistive technologies for digital inclusion. AI leverages computer vision models to comprehend visuals, and natural language processing (NLP) models to mimic human writing and present information understandably; the natural result of applying these systems is auto-generated captions. As hardware becomes more prominent and powerful, AI methods expand in size and complexity to fill the space and utilize as many resources as possible. Still, the monetary cost of maintaining and running those models is incompatible with the goal of accessibility. The problem is challenging to solve. There is not any money to be made in increasing digital inclusion and accessibility with open-source tools, so the private sector doesn’t push for solutions, the federal government cannot sue the entire internet to enforce ADA compliance, and the race to develop state-of-the-art machine learning models means researchers aren’t particularly concerned with taking time to engineer systems for their existing research methods. Aspiring data scientists are left with inadequate tools and insufficient knowledge to bridge the gap between machine learning models and accessible assistive technology.

The team developed a machine learning-powered pipeline that, given an image, describes its contents in detail so that a person can understand it without seeing. The image's description includes objects, relative location, text, and background of the picture.

## Data Flow
The data flow indicates the process of steps of which our data progresses through the pipeline that we built. In essence, an image is input into the pipeline by the user, which then goes through 3 initial computer vision models: Object Detection, Scene Recognition, and Optical Character Recognition. Once the Object Detection model is deployed, the Human Activity Recognition model is given permission to run, a form of conditionality only set for these two models. This is important because we do not want any human activities to be looked at if there are no objects present in the first place. Once these four models have completed their task, they are sent to the NLG model to form a caption. Then, the outputs are organized: location of objects, the caption from NLG, and the text pulled from OCR. Finally, the text output is given to our text to speech model and will be audible for the user. This pipeline’s data flow is extremely important because it is the only logical way for the steps to progress. For example, it does not make sense for the NLG model to start building a caption without the outputs from our CV models. The conditionalities were built in our code using a series of if statements to allow each model to know when the prior was running.

## Datasets for Computer Vision and Natural Language Generation

### Common Gen
- Introduced in a research paper in the Association for Computational Linguistics
- The file is in JSON Format and available in test, train, validation split
- Hosted and held in Hugging Face, the dataset includes three fields – concept_set_idx, concept, target
### Common Objects in Context 2017
- Used for Object Detection and Natural Language Processing
- Stored in JPEG Format for Object detection, includes full-color images sourced from diverse locations on the internet
- The Image caption dataset, integral to NLP model training, contains annotations in JSON format
### Stanford 40
- 9532 images with 40 Classes of Human Actions
- Stored in JPEG Format for action recognition with full-color images
- The current hosting site does not specify how the images were acquired or the criteria of the classes
### SUN 397
- SUN (Scene Understanding) 397 dataset contains 108,754 still images of scenes. 397 classes of scenes with 100 images per scene
- Due to the large size of the dataset, we used one-quarter of the dataset
- Does not contain annotations, instead organized into folders that denote the label of the images
  
## Computer Vision

### Object Recognition:
This is the first step in the image captioning pipeline, which detects what items, people, or animals appear in the image.
This module contains functions for object detection using YOLOv8 (base model, size X). They will accept input in the form of an open image, run them through YOLOv8, and parse the Ultralytics results to deliver a dictionary of objects with class names, confidence scores, xyxy bounding boxes. If the 'bounding box' option is checked in Streamlit, it will also write a version of the input image to file with the bounding boxes to be displayed by Streamlit. This module also uses the location module to add absolute location and centroid information to each object dictionary for use in downstream tasks. The confidence level can be adjusted up or down to filter the results according to the class confidence returned by YOLOv8.

Results can be further output as a list of strings of class names (with or without metadata) for use in natural language generation.
If the yolo8x.pt file is not present, the Ultralytics package will automatically download it locally for use before proceeding.

### Scene Recognition:
This module is the second independent step in the pipeline after object detection, and will return a class label for the predicted setting of the image.  It accepts the dictionary output of the object detection stage and the opened image as inputs.
This module uses a custom-fine-tuned Densenet201 model created using Pytorch Torchvision and ImageNet default weights from that module, further trained using a quarter sample of the SUN 397 scene dataset with the Averaged Stochastic Gradient Descent (ASGD) optimizer.  This custom model is 'densenet_asgd_best.pt', and the module will not function without it. The module will run on GPU or CPU in that priority order. It also contains necessary transforms (convert to RGB, 256x256 image resizing, center crop, and standard RGB normalization) for image pre-processing, as well as functions to apply human readable labels to the predicted class and add that label to the image dictionary and list of strings created in the object detection phase.

### Location:
This module provides a number of different utility functions related to the location of objects within an image.  This information is added to objects in the object detection stage and is also used to assign actions to objects in the action recognition stage.
Functions in this module, which are calculated using only Numpy and math, include: finding the centroid of a bounding box in xyxy format, finding the Euclidean distance between two bounding box centroids, and finding the absolute and relative location of bounding boxes. 
For absolute location, the module divides the image given into 9 boxes and determines which of the 9 boxes a given bounding box's centroid falls within. The box is then assigned a descriptive phrase (such as 'middle left') which indicates its absolute location.
For relative location, given a pair of bounding boxes, the module will calculate the angle of a line drawn between the two centroids, and assign a descriptive phrase (such as 'above') that indicates the position of the first bounding box relative to the second bounding box.

### Human Actions:
Generally, captions fall into the form object-action-context, so it was important to include a computer vision model for human activity recognition (HAR) to detect actions. Literature on the subject is heavily skewed towards large models processing video data, though, and the current project focuses on lightweight models for still-image photo captioning. Lacking other resources to build our desired pipeline, we retrained YoloV8 by ultralytics on the Stanford 40 Actions dataset to become a simple object detection model for HAR.

To implement the HAR model “yolo_act.pt”, found in the root of the directory, the functions in “\actions\yolo_act.py” were used. standalone_yolo_act() calls and runs the model “yolo_act.pt” and returns a dictionary of detected labels, their bounding boxes, and the location of their centroids. Then, assign(), calculates the optimal pairing of actions to “person” labels from the results of the object detection model. It creates a list of every possible permutation of pairings then sums the Euclidean distance between the pairings within that permutation; the permutation with the minimum summed distance is considered the most optimal pairing. The pairings outputted from assign() are then called within the function action_assign(), where the object detection dictionary is updated to include the action labels assigned to the proper person. The above functions are wrapped succinctly into run_yolo_act() and called by streamlit in the notebook “\streamlit_modules\page_1.py”. The function output_class_list_with_actions() is then called as a method of outputting the labels from the dictionary with the action assignments in a manner that eliminates any underscores for spacing so that the text is ready to be put through the Natural Language Generation model.

### OCR:
Optical Character Recognition is a form of computer vision that pulls the text from an image. This is an important form of computer vision for our pipeline because it is extremely powerful in building the context for the target audience. Since there are numerous options available consisting of strong and lightweight models, this feature was easily implemented into the pipeline’s data flow. Code implementation required both the pytorch library and other basic computer vision libraries to implement any form of image inputting and extraction of text from jpeg images. Though multiple models were assessed, Keras_OCR was the most lightweight and accurate model that fit our pipeline’s guidelines of focusing on the lowest time to completion.

To place the words detected by OCR in the proper order, “\ocr\ocr.py” contains the rotate() and order_of_words() functions. rotate() calculates the angle of rotation such that the top left and top right vertices of the bounding box of the longest word have the same y-coordinate value and would therefore be horizontal. This rotation is then applied to all the bounding box vertices for the words detected and passed along as a newly transformed list of arrays for order_of_words(). order_of_words steps through the bounding boxes of the detected words and establishes which words are in line. The words that are in line are then organized from left to right so the output can be read in the proper order.

## Natural Language Generation
The T5 is the model used for the NLP/NLG module in the engine. The T5 is a transformer so it can receive text as an input and release text as an output as well. T5 can be used for summarization, translation and generation. This variation of the T5 was pretrained on the common gen dataset, which focused on text generation. The format of the dataset includes one field as keywords and another field as sentences generated from the keywords. This was very useful for this report as keywords were received from the computer vision models and a common sense sentence was expected to be generated. The pre trained model is available on HuggingFace. Therefore, installing the packages and libraries to access the model is all that is needed to run the model.   

## Streamlit Module

### Setup

#### What Streamlit Module Affects

The Streamlit Module is crucial for the overall functionality of our application. It directly impacts the user interface and experience by shaping how users interact with our machine-learning models. Its design and implementation directly influence the usability and efficiency of the data application.

### Setup Requirements

To set up the Streamlit Module, you'll need to install Python and the Streamlit library on your machine. Depending on the specific functionality of your application, you may also need additional Python libraries for data processing, model implementation, and visualization.

### Beginning with Streamlit Module

To get started with the Streamlit Module, clone our project repository, pip or conda install requirements.txt and navigate to the streamlit_modules folder. From there, you can examine and run the individual Streamlit scripts to see how each application part works.

### Usage

The streamlit_modules don't need to be configured for usage. If the directory is in the same path as the main.py script, the Streamlit application should run as intended. The three-page scripts (page_1.py, page_2.py, page_3.py) can be modified to change the content of their respective pages. The module works as follows:

- **Import Necessary Libraries and Modules:** The necessary Python libraries and other modules needed for running the models are imported.

\```

pip install -r requirements.txt

\```

- **Run the Streamlit app locally:**

\```

streamlit run main.py

\```

- **User Input:** Use the sidebar to select Model testing or page 1, then select yolo3 to see the model we started with or yolo8 to see our final product, then under select NLP Models, select the T5_COCO(Baby T5) to see our attempt at retraining Googles T5 on the COCO Captions dataset which had marginal performance or select T5 Common GEN to see the best performance from a pre-trained model. Browse and upload an image, move the confidence slider to your desired confidence level (50 to 70% performs best), then select the yes or no radial button if you would like bounding boxes displayed or not, then click run models, then scroll down to click the text to speech description of the image.

- **Model Implementation:** Once the user input is provided, the relevant model or models are implemented using the input user input data.

- **Output Display:** The model results are displayed back to the user in a user-friendly format, which includes the output from the Densenet scene recognition model, The yolo Object detection models, the yolo8 retrained for action detection is run if yolo8 objects detect a person, Then the computer vision labels are sent to a preprocessing function to prepare them to input into the 2x optional T5 models. The yolo image with bounding box and confidence levels are displayed, then all of the model components that feed the T5 are displayed, then Location and OCR information and a smaller OCR photo with bounding boxes are displayed separately. The text-to-speech sound bar is displayed at the bottom.

### Limitations

The Streamlit library, although an excellent tool for developing data science applications, does come with a few limitations:

- **Resource Limits:** The Streamlit Community Cloud for hosting has a 1GB resource limit, which includes memory (RAM), CPU, and disk storage. We mitigated this issue for our project by hosting our application on an AWS EC2 instance with a 100GB SSD.

- **Cache Limitations:** Streamlit's caching mechanism can't track changes to the data outside the function body. It only checks for modifications within the current working directory. Streamlit cache and cache resource functions were not helpful for running our models.

- **Default Themes:** Streamlit currently offers a limited selection of default themes for the applications, but they have some functionality for customizing themes [here](https://blog.streamlit.io/introducing-theming/). But for more robust control, something like Flask might be better.

- **Lack of Voice-to-Text Support:** Initially, we intended to incorporate a voice-command feature for the visually impaired. However, Streamlit doesn't support any libraries that allow voice-to-text commands.

### Development

Contributions to the Streamlit Module are welcome. To get involved, please fork the project repository, and make your changes in a new branch. You can open a pull request once you're ready to share your contributions.

