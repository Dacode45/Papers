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