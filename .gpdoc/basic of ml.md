# Machine Learn Basics

Machine learning is a field of artificial intelligence that enables systems to learn from data without being explicitly programmed. Here are some basic concepts explained with code examples:

### 1. Supervised Learning:
In supervised learning, the algorithm learns from labeled data, where each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal).

```python
# Example of supervised learning using scikit-learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)
```

### 2. Unsupervised Learning:
In unsupervised learning, the algorithm learns patterns from unlabeled data. It explores the data and can find hidden structures or patterns.

```python
# Example of unsupervised learning using scikit-learn
from sklearn.cluster import KMeans
import numpy as np

# Generate random data points
X = np.random.rand(100, 2)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_
```

### 3. Reinforcement Learning:
Reinforcement learning is about taking suitable actions to maximize reward in a particular situation. The learner learns to achieve a goal in an uncertain, potentially complex environment.

```python
# Example of reinforcement learning using OpenAI Gym
import gym

# Create an environment
env = gym.make('CartPole-v1')

# Run episodes with random actions
for _ in range(10):
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
```

These examples showcase the core concepts of machine learning through code snippets in Python using popular libraries like scikit-learn and OpenAI Gym.