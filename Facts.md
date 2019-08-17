# Some Important questions discovered in ML
---
## Why L1 loss preferred for sparse models?

> With a sparse model, we think of a model where many of the weights are 0. Let us therefore reason about how L1-regularization is more likely to create 0-weights.
Note that L2-regularization can make a weight reach zero if the step size ηη is so high that it reaches zero in a single step. Even if L2-regularization on its own over or undershoots 0, it can still reach a weight of 0 when used together with an objective function that tries to minimize the error of the model with respect to the weights. In that case, finding the best weights of the model is a trade-off between regularizing (having small weights) and minimizing loss (fitting the training data), and the result of that trade-off can be that the best value for some weights are 0.

## Various loss function for regression problems?
--- 
- Mean square error/L2
- Mean absolute error/L1
- Mean absolute Percentage error
- Quantile loss
- Log cosh loss 
 
## Loss for classification?
---
- Hinge loss
- Log loss
- Exponential loss

## Compare L1 and L2 loss?
---
- In short, using the squared error is easier to solve, but using the absolute error is more robust to outliers.
- Gradient of L1 loss is same throughout.Even for smaller loss gradient is high.However it can be fixed by dynamic learning rate
 
## Problems with L1 and L2 loss?
> One big problem with using MAE for training of neural nets is its constantly large gradient, which can lead to missing minima at the end of training using gradient descent. For MSE, gradient decreases as the loss gets close to its minima, making it more precise

## Tell about Huber loss and why it is better than both l1 and l2 loss?
> Huber loss is less sensitive to outliers in data than the squared error loss. It’s also differentiable at 0. It’s basically absolute error, which becomes quadratic when error is small.

## Significance of weights and biases in ANN?
> The weight shows the effectiveness of a particular input. More the weight of input, more it will have impact on network.
On the other hand Bias is like the intercept added in a linear equation. It is an additional parameter in the Neural Network which is used to adjust the output along with the weighted sum of the inputs to the neuron. Therefore Bias is a constant which helps the model in a way that it can fit best for the given data.

## Different activation function with formulas and their limitations?
---
#### Sigmoid Function
>f(x) = 1 / 1 + exp(-x) 

#### Problems:
- Vanishing gradient problem
- Secondly , its output isn’t zero centered. It makes the gradient updates go too far in different directions. 0 < output < 1, and it makes optimization harder.
- Sigmoids saturate and kill gradients.
- Sigmoids have slow convergence

#### Tanh function

> f(x) = 1 — exp(-2x) / 1 + exp(-2x) 

#### Problems:
- Vanishing Gradient problems.

#### Relu function
> R(x) = max(0,x) i.e if x < 0 , R(x) = 0 and if x >= 0 , R(x) = x. 

Rectifies vanishing gradient problem.
#### Problems:
- Used with hidden layers only.
- Another problem with ReLu is that some gradients can be fragile during training and can die. It can cause a weight update which will makes it never activate on any data point again. Simply saying that ReLu could result in Dead Neurons.
- 
These problems were solved by leaky relu function.

## Xgboost characteristics and tuning params(generally tuned)
---
XGBoost is one of the fastest implementations of gradient boosted trees.
It does this by tackling one of the major inefficiencies of gradient boosted trees: considering the potential loss for all possible splits to create a new branch (especially if you consider the case where there are thousands of features, and therefore thousands of possible splits). XGBoost tackles this inefficiency by looking at the distribution of features across all data points in a leaf and using this information to reduce the search space of possible feature splits.
The params that can be tuned are:
- N_estimaters
- Max_depth: ( value =5 is self sufficient for even complex dataset)
- Learning_rate
- Reg_alpha, reg_lambda

## Define 4 plot and its significance:
---
Some basic plot for any ml model to check:
- **Lag plot:** for randomness
- **Run sequence plot:** to check fixed location of values
- **Histogram:** for testing normal distribution
- **Normal probability plot:** for testing normal distribution

#### Inference results
- If the fixed location assumption holds, then the run sequence plot will be flat and non-drifting.
- If the fixed variation assumption holds, then the vertical spread in the run sequence plot will be approximately the same over the entire horizontal axis.
- If the randomness assumption holds, then the lag plot will be structureless and random.
- If the fixed distribution assumption holds (in particular, if the fixed normal distribution assumption holds), then the histogram will be bell-shaped and the normal probability plot will be approximatelylinear.

If all 4 of the assumptions hold, then the process is "statistically in control". In practice, many processes fall short of achieving this ideal.

## Some visualization plots for different dimensional data:
---
### 1-D data
	1. bar plot: count and frequency
	2. box plot: descriptive statistics
	3. kdeplot: distribution
	4. histogram: distribution
	5. line graphs: trend of values
	6. pie chart: categorical distribution/population
	7. scatter plot: range of values
    8. Autocorrelation plot: for randomneess
	 
### 2-D data
	1. Scatter plot
	2. pairplot
	3. parrallelcordindates plot
	4. bubble plot
	5. violin plot
	6. box plot
	7.joint pot

## Distinguishing feature for vanishing/exploding gradient problems
---
A good standard deviation for the activations is on the order of 0.5 to 2.0. Significantly outside of this range may indicate vanishing or exploding activations.


## Types of anomaly detection
---
Anomalies or outliers come in three types.
- **Point Anomalies:** If an individual data instance can be considered as anomalous with respect to the rest of the data (e.g. purchase with large transaction value)
- **Contextual Anomalies:** If a data instance is anomalous in a specific context, but not otherwise ( anomaly if occur at a certain time or a certain region. e.g. large spike at the middle of the night)
- **Collective Anomalies:** If a collection of related data instances is anomalous with respect to the entire dataset, but not individual values. They have two variations.
        1. Events in unexpected order ( ordered. e.g. breaking rhythm in ECG)
        2.	Unexpected value combinations ( unordered. e.g. buying a large number of expensive items)

![image](https://user-images.githubusercontent.com/31818185/63217127-0070f480-c15e-11e9-8d94-5617debdaae8.png)

 
## Causes of outlier
---
- Data entry errors (human errors)
- Measurement errors (instrument errors)
- Experimental errors (data extraction or experiment planning/executing errors)
- Intentional (dummy outliers made to test detection methods)
- Data processing errors (data manipulation or data set unintended mutations)
- Sampling errors (extracting or mixing data from wrong or various sources)
- Natural (not an error, novelties in data)

## Methods for outlier detection?
---
- Z-Score or Extreme Value Analysis (parametric)
- Probabilistic and Statistical Modeling (parametric)
- Linear Regression Models (PCA, LMS)
- Proximity Based Models (non-parametric)
- Information Theory Models
- High Dimensional Outlier Detection Methods (high dimensional sparse data)



