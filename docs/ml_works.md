--- 
layout: default
description: "How does Machine Learning Work?"
---
<h1>How Does Machine Learning Work?</h1>

<img src="https://uploads-ssl.webflow.com/5bfc592598ce06d4cf4fc47d/5cd46fdd63a1fe6804697e90_AI-MachineLearning-DeepLearning-Relationship.png" alt="Chart describing how machine learning is a subset of AI" style="height: 400px;"><br>
Courtesy of Sigma IQ; The difference and similarities of AI and machine learning.

**Machine Learning is a subset of Artificial Intelligence (AI), which is used in things like Alexa and Google Assistant. AI tries to give machines human-level intelligence. Machine learning, however, is only used for one purpose at a time, so it does not try to achieve human-level intelligence. Machine learning has specific specializations like sorting pictures, organizing data, or translating data to graphs.**

**In Waggle Dance, I use machine learning to classify different species of bees. There are several different tools to make machine learning more accessible, like [Tensorflow](https://www.tensorflow.org/), which I used. Once you have the machine learning tool, you have to make the algorithm, which is about as hard as it sounds. Tensorflow has a multitude of tutorials which helped me achieve my goal, along with assistance from mentors.**

**When the machine learning algorithm is finished, you then move on to training the model. To train a classifier model like mine, a binary model, which will match any picture even if it is not a bee, you need thousands of labelled pictures, which teaches the model what the difference between your data is, like the difference between honey bees and bumblebees. Luckily for me, a [METIS challenge](https://www.drivendata.org/competitions/8/) had already gathered labelled training data.**

**After your model is trained, it has to be tested to find the accuracy. That needs tons more labeled data so it can find the true accuracy of the model. A lot of models suffer from overfitting, which is when a model learns the specifics of its training data, and not what the true difference between two things are. One of the ways to fix overfitting is getting more data, which can be done by generating data out of labeled data you already have. You can generate data in a multitude of ways, like cropping, zooming, rotating, shifts, and flips.**

<img src="./augmented_im.png" alt="Examples of solutions to overfitting" style="height: 400px;"><br>
Examples of solutions to overfitting.

**After generating more data, overfitting should be mostly solved, and there are most likely few more things that you have to do. There are tons of good resources that can help beginners in machine learning, so it is very easy to start from your home laptop!**

<img src="https://www.sqlservercentral.com/wp-content/uploads/legacy/60fdeff7bbe5926b57a764b9772b7b55636c32fc/34296.png" alt="Machine Learning flow chart" style="height: 300px;"><br>
Courtesy of SQL ServerCentral; A flowchart for the process of machine learning.

**Here is a list of the different resources that I used:**
* **https://github.com/chetannaik/bee_classifier_using_cnn**
* **https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb**
* **https://www.tensorflow.org/tutorials/images/classification**
