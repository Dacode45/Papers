## 4.1.1 Supervised Learning
**Supervised Learning**: Mapping input data to known targets (called *annotations*).
Common Types:
- Sequence generation - Given a picture predict a caption. Can be considered a series of classifcation problems (such as repeatedly predicting a word or token in a sequence)
- Syntax tree prediction - Given a sentance, predict its decomposition into a syntax tree
- Object detection - Given a picture draw a bounding box around it. It can also be a classification problem
- Image segmentation - Given a picture, draw a pixel-level mask on a speccific object.

## 4.1.2 Unsupervised Learning
Find interesting transformations of input without the help of targets.
Common Types:
- Dimensionality reduction
- Clustering

## 4.1.3 Self-supervised Learning
No human involvment. Get labels from the input data using some sort of heuristic.
Common Types:
- Autoencoders
- Predict next frame in a video, given past frames.

## 4.1.4 Reinforcement Learning

# 4.2 Evaluating machine-learning models
Lets review evaluation recipies

#### Simple Hold-out Validation
![](/assets/Screenshot from 2018-06-07 12-08-03.png)



```
num_validation_samples = 10000
np.random.shuffle(data) # Shuffle data

validation_data = data[:num_validation_samples]

training_data = data[:]

model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# At this point we can tune model, retrain, evaluate, and tun again

model = get_model()
# Once hyperparameters are tuned, it's common to train your final
# model from scratch on all non-test data available.
model.train(np.concatenate([training_data, validation_data]))

test_score = model.evaluate(test_data)
```

This method has a fata flaw: if little data is available, then the validation and test sets may contain too few samples to be statistcally representative.

We can recognize this easily: if different random shuffling rounds before splitting end up yielding very different measures of model performance, then we have this issue. We can address this with K-fold validation

#### K-fold Validation

Spit data into K sets. For every ith set train on the remaing K-1 sets. Average the final score.

![](/assets/Screenshot from 2018-06-07 12-19-02.png)



```
k = 4
num_validation_samples = len(data) // k

np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[num_validation_samples * fold: num_validation_samples * (fold + 1)]
    training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
    
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_scores)
    
validation_score = np.average(validation_scores)

# train the final model on all non test data vailable
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)
```

#### Iterated K-Fold Validation with Shuffling
This is for when there is little data and you need to evaluate your model as precisely as possible. Useful for Kagle. Apply K-fold multiple times. Shuffling the data every time before splitting K ways. The final score is the average of each run of K-fold validation. This means P*K (where P is the number of iterations) models are trained and evaluated.

### Things to keep in mind
- Data representativeness - Make sure training and test sets are representative. A common mistake is data being ordered by class. The split for training and test will cause classes to be left out in either one. For this reason it's good to shuffle before splitting.
- The arrow of time - If you're trying to predict the future given the past, do not randomly suffle the data before splitting. Make sure all data in test set is posterior to the data in the training set.
- Redundancy in your data - Some data points appear twice in real world data. Ensure thaat you keep the training set and validation set disjoint.

# Data preprocessing, feature engineering, and feature learning


