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
How to prepare data before feeding them into a neural network

## 4.3.1 Data preprocessing for neural networks
Makes raw data more amenable to nural networks: vectorization, normalization, handling missing values, and feature extraction

### Vectorization
Convert inputs and targets to tensors of floating-point data
### Value Normalization
In the digit-classification, we start with image data encoded as integers in the 0-255 range, endcoding grayscale values. We had to cast that to a float32 and divide by 255 to have floating point in 0-1 range.

In the predicting house prices example, each feature had to be normalized independently so that it had a standard deviation of 1 and a mean of 0.

Attempt to maintain the following
- Take small values - Most values hould be in the 0-1 range.
- Be homogenious - All featues should take values in roughly the same range.

A stricter practice is to force each feature to have a mean of 0 and a std of 1.
```
# Assuming x is a 2D data matrix
x -= x.mean(axis=0)
x /= x.std(axis=0)
```

For most networks it is safe to input missing values as 0 as long as 0 isn't already a meaningful value. The network will reallize that 0 means missing data and will ignore the value.

Note that if you are expecting missing values in the test data, but the network was trained on data without missing values, the network won't have learnind to ignore missing alues!

### Feature engineering
Transform data to make it easy for the network. Instead of a complete image, transform it into a simple number.

# Overfitting and Underfitting

## 4.4.1 Reduce the network's size
The simplest way to prevent overfitting is to reduce the size of the model: the number of learnable parameters in the model (which is determined by the number of layers and the number of units perr layer). 

In DL, the number of learnable parameters in a model is the model's capacity. A model with more parameters has more **memorizaiton capacity** and therefore can easily learn a perfect dictionary-like mapping between training a target. A model with 500,000 binary parameters could learn the class of every digit in MNIST. We'd only need 10 binary parameters for each of the 50,000 digits.

There's no formula to do this. We have to evaluate an array of different architectures. Lets see and example with the movie-review classification network

```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
NOw replace it with a smaller network
```
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
![](/assets/Screenshot from 2018-06-07 12-55-33.png)

The smaller network overfits later.

For kicks, lets add a network with more capcity than the problem warrants.

```python
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

The validation loss of the larger model kicks in immediately after 1 epoc but the training loss goes down fast.

## Adding weight regularization
We can penalize complexity forcing the model to take on weights with smaller values. This makes the distribution of weights more regular. This is called *weight regularization*, and it's done by adding to the loss function of the network a cost associated with having large weights.
- *L1 regularization* - the cost is proportional to the *absolute value of the weight coefficients* (the L1 norm of the weights).
- *L2 regulairzation* - the cost is proportional to the *sqare of the value of the wieght coefficients* (the L2 *norm* of the weights). L2 regularization is called weight decay.

We can do this in Keras by passing wieght regularizer instances to layers.

```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation = 'relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

0.001 * weight_coefficient_value will be added to the total loss of the netowrk. This is only added at training time.

The effects are rather significant.
![](/assets/Screenshot from 2018-06-07 13-05-14.png)

Keras has a couble regularizers
```python
from keras import regularizers
# L1 regularization
regularizers.l1(0.001)

# L1 + L2 regularization
regularizers.l1_l2(l1=0.001, l2=0.001)
```

## Adding dropout
Dropout consists of randomly dropping out (setting to zero) a number of output features during training. Let's say a layer would normally return a vector `[0.2, 0.5, 1.3, 0.8, 1.1]` for a given input. After dropout the vector will have a few zero entries at random : `[0, 0.5, 1.3, 0, 1.1]`. The dropout rate is the fraction of the features that are zeroed out; it's usually set between 0.2 and 0.5. At test time no units are dropped out; instead, the layer's output values are scaled down by a factor equal to the dropout rate, to balance for the fact that more units are active than at training.

Consider a Numpy matrix containing the output of a layer, layer_output, of shape (batch_size, features). We zero out a random fraction in this matrix
```python
# drop out 50% of the units.
layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
```
At test time we scale by the dropout rate
```python
layer_output *= 0.5
```

Hinton, who created this method, theorized that randomly removing a different subset of neurons on each example would prevent conspiracies and thus reduce overfitting.

We do this in Keras by the following
```python
model.add(layers.Dropout(0.5))
```

Lets add two to the IMDB network

```python
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
```

# 4.5 the universal workflow of machine learning
Here's the blueprint to sovle these problems

## 4.5.1 Defining the problem and assemling a dataset
- What is the input data? What are you trying to predict?
- What problem are you facing? Binary classifcation? Multiclass classifcation? Scalar regression? Vector regression?

Remember what your hypotheses are
- You hypothesize that your outputs can be predicted given your inputs.
- You hypothesize that your available data is sufficently informative to learn the relationship between inputs and outputs

## 4.5.2 Choosing a measure of success
For balanced-classification problems, where every class is equally likely, accuracy and *area under the reciever operating characteristic curve* (ROC UC) are common metrics. For imbalznced problems you can use precision and recal. For ranking or multilabel, use mean average precision.

Kaggle showcases a wide range of problems and evaluation metrics

## 4.5.3 Deciding on an evaluation protocol. 
- Maintain hold out validation set
- K-fold
- iterated k-fold

## 4.5.4 Prepare data
## 4.5.5 Develop model that does better than a baseline.
Does your model do better than deciding at random?

There are three key choices to build your first working model
- Last-layer activation - This establishes useful constraints on the network's output. For instance, the IMDB classification example used sigmoid in the last layer; the regression example didn't use any last-layer activation; and so on.
- Loss function - This should match the type of problem you're trying to solve. For instance IMDB used binary_crossentropy, the regression used mse
- Optimization configuration - What optimizer and learning rate? It's save to go with rmsprop and its default learning rate.

![](/assets/Screenshot from 2018-06-07 13-24-42.png)

## Develop a model that overfits
The goal is to distinguish between optimization and generalization. To figure out the border cross it first.

- Add layers
- Make the layers bigger
- Train for more epocs.

Monitor the trainign loss and validation loss, as well as the traning and validation values for any metrics you care about. Once performance degrades, we've overfit.

The next stage is regularzation and tuning the model.

## Regularizing your model and tuning your hyperparameters

- Add dropout
- Try differennt architectues
- Add L1/L2 regularization
- Try different hyperparameters (units per layer or learning rate)
- Trye featue engineering.