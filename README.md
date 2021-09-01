# CSE_881

## ABSTRACT
Face recognition is a hot topic in machine learning. It is useful in many situations. Expression recognition is also a useful sub-field and have plenty of practical use. In this project, the team trained traditional classifiers and convolutional neural network on CK+ dataset and tested on AffectNet. Results have shown even a simple convolutional neural network could result in high training accuracy, but the accuracy on test set is poor. The traditional classifiers did not perform well on training nor test data.

## INTRODUCTION
Recognizing facial expression comes in natural for humans, but it is not necessarily an easy task for machine. It is challenging for machine because such task often involves high dimensional space. It the images are 50 × 50, then the dimension is already at a whooping 2500. There could be many potential application for such automatic expression detector. It could be used in lie detector to monitor change in micro expression, be implemented on online psychological consulting service to detect the status of the subject, and be applied in public safety to identify potential perpetrator a head of time. This project is an endeavor to develop classifiers to recognize human facial expression. The models developed did not perform well on the test set; nevertheless, over the course of this project, the team made great contribution and learned many frameworks, classifiers, and preprocessing techniques.

## CONCLUSION
Despite the effort, this project was not able to produce a good model for real life use based on posed expression. The results have a num- ber of implications. First of all, to recognize natural expression, the models might have to be trained on natural faces. Secondly, the performance of the models severely deteriorates on the test set, suggesting the models might be trained in a way to recognize the statistical attributes of the dataset instead of the expressions. Last but not least, the size of the training set might not be ade- quate to achieve the task. Even though the project itself was not successful, there are important lessons learned. KNN has proven not a good classifier for expression recognition, and perhaps this conclusion could be extended to image classification in general. It is possible that certain preprocessing technique does not help training performance but improves performance on the test set; in this case Canny edge detection helped the models overall in noisier environment than training set. CNN remains a strong contender in image classification; however, when the settings of the test set vary significantly, the performance will suffer.
