# Exploring-Supersymmetry-with-Machine-Learning
# Machine Learning Scan for Exploring the CMSSM Parameter Space
Machine Learning Scan (MLS) is a specific application of machine learning tailored for the exploration of high-dimensional parameter spaces in theoretical physics, particularly for models like the Minimal Supersymmetric Standard Model (MSSM) and the Constrained MSSM (CMSSM). Here's a detailed look at the architecture and structure of MLS.
## Steps and Concepts

### 1. **Parameter Space and Initial Sampling**

The parameter space for the Constrained Minimal Supersymmetric Standard Model (CMSSM) typically includes:
- $M_0$: Universal scalar mass (range: [50, 2000] GeV)
- $M_{1/2}$: Universal gaugino mass (range: [50, 3000] GeV)
- $A_0$: Universal trilinear coupling (range: [-3, 3])
- $\tan\beta$: Ratio of Higgs vacuum expectation values (range: [3, 50])
- $\mu$: Higgsino mass parameter (range: [50, 2000] GeV)

**Initial Sampling**: 
Generate an initial set of random samples uniformly distributed across the specified ranges for each parameter. This ensures a broad coverage of the parameter space.

```python
import numpy as np

def initial_sampling(num_samples, param_ranges):
    return np.random.uniform(param_ranges[:, 0], param_ranges[:, 1], size=(num_samples, param_ranges.shape[0]))

# Define parameter ranges
param_ranges = np.array([
    [50, 2000],   # M0
    [50, 3000],   # M1/2
    [-3, 3],      # A0
    [3, 50],      # tan(beta)
    [50, 2000]    # mu
])

# Generate initial samples
initial_num_samples = 100
params = initial_sampling(initial_num_samples, param_ranges)
```

### 2. **Physical Observables**

Physical observables are quantities predicted by the CMSSM that can be compared to experimental data. Examples include:
- Mass of the Higgs boson $m_h$
- Dark matter relic density $\Omega_{\text{DM}} h^2$
- Supersymmetric particle masses (e.g., neutralinos, charginos)
- Electroweak precision observables
- Flavor physics observables

These observables are computed using sophisticated high-energy physics (HEP) packages like `micrOMEGAs`, `SoftSUSY`, `FeynHiggs`, etc.

### 3. **Supervised Learning and Labeled Data**

**Labeled Data**:
- **Features**: The CMSSM parameter sets (e.g., $M_0$, $M_{1/2}$, $A_0$, $\tan\beta$, $\mu$ )
- **Labels**: Corresponding physical observables computed for these parameter sets

The initial set of parameter samples is used to compute the physical observables, forming the labeled dataset for supervised learning.

```python
# Placeholder function to calculate physical observables
# In practice, this would use HEP packages to calculate observables for each parameter set
def calculate_observables(params):
    observables = []
    for p in params:
        # Simulate calculation of observables (e.g., Higgs mass, relic density)
        observable = physical_model(p)
        observables.append(observable)
    return np.array(observables)

# Calculate observables for the initial samples
observables = calculate_observables(params)

# Labeled data for supervised learning
X_train = params
y_train = observables
```
### 4. **Training the Machine Learning Model**
Machine Learning Models:

Deep Fully Connected Neural Networks: The default model used in MLS. These networks have multiple hidden layers with rectified linear unit (ReLU) activation functions and are optimized using the Adam optimizer.
Incremental and Active Learning: Continuously update the model with new data points and actively focus on regions of interest based on the reconstructed likelihood.
Train a machine learning model (e.g., a deep neural network) to predict physical observables based on the parameter sets.

```python
from sklearn.neural_network import MLPRegressor

def train_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train the model
model = train_model(X_train, y_train)
```



### 5. **Guided Sampling**
Guided sampling is a critical part of the MLS workflow, where the goal is to intelligently select new parameter points to sample, focusing on those that are likely to yield high likelihood values. This process is balanced by also including some random sampling to ensure comprehensive exploration of the parameter space.
Use the trained model to guide the selection of new parameter points to sample, focusing on regions of high likelihood (where the model predictions are most promising).

**Guided Sampling**

1. **Generate New Candidate Parameter Points**: 
   Generate new points in the parameter space.

2. **Predict Physical Observables**: 
   Use the trained model to predict the physical observables for these new points.

3. **Compute Likelihood**: 
   Compute the likelihood of each candidate point based on the predicted observables.

4. **Select High-Likelihood Points**: 
   Select points with high likelihood values to be included in the new sample set.

5. **Include Random Points**: 
   Additionally, include a few completely random points to ensure the model explores the entire parameter space and avoids local optima.

 **Likelihood Calculation**

The likelihood function is used to evaluate how well the predicted observables match the target values (experimental data). Points with higher likelihood are more likely to be included in the new sample set.


```python
def guided_sampling(model, param_ranges, num_samples, num_random_points):
    new_samples = []
    while len(new_samples) < num_samples - num_random_points:
        sample = np.random.uniform(param_ranges[:, 0], param_ranges[:, 1], param_ranges.shape[0])
        pred_obs = model.predict(sample.reshape(1, -1))
        if np.random.rand() < likelihood(2, pred_obs, 0.1):  # Placeholder likelihood calculation
            new_samples.append(sample)
    random_samples = np.random.uniform(param_ranges[:, 0], param_ranges[:, 1], (num_random_points, param_ranges.shape[0]))
    return np.vstack((new_samples, random_samples))

# Perform guided sampling
new_params = guided_sampling(model, param_ranges, num_new_samples=50, num_random_points=10)
```
### 6. **Observable Calculation and Data Integration**

Calculate the physical observables for the newly sampled points and integrate them into the training set.

```python
# Calculate observables for the new samples
new_observables = calculate_observables(new_params)

# Update training data
X_train = np.vstack((X_train, new_params))
y_train = np.vstack((y_train, new_observables))
```
### 7. **Iteration**

Repeat the process iteratively, improving the model and refining the parameter space exploration.

```python
iterations = 10

for iteration in range(iterations):
    print(f"Iteration {iteration + 1}")

    # Train model
    model = train_model(X_train, y_train)

    # Guided sampling
    new_params = guided_sampling(model, param_ranges, num_new_samples=50, num_random_points=10)
    new_observables = calculate_observables(new_params)

    # Update training data
    X_train = np.vstack((X_train, new_params))
    y_train = np.vstack((y_train, new_observables))

    # Evaluate model performance
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    print(f"Mean Squared Error: {mse}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.colorbar(label='Observable')
plt.xlabel('M0 (GeV)')
plt.ylabel('M1/2 (GeV)')
plt.title('MLS Sampled Parameter Space')
plt.grid(True)
plt.show()

```
### Summary

- **Parameter Space**: Defined by $M_0$, $M_{1/2}$, $A_0$, $\tan\beta$, and $\mu$.
- **Initial Sampling**: Generate random samples within specified ranges.
- **Physical Observables**: Quantities predicted by CMSSM, calculated using HEP packages.
- **Labeled Data**: Parameter sets (features) and corresponding physical observables (labels).
- **Guided Sampling**: Use the trained model to focus sampling on high-likelihood regions.
- **Iterative Process**: Repeat training, sampling, and data integration to improve the model and explore the parameter space efficiently.

By following this workflow, the Machine Learning Scan (MLS) can efficiently explore the high-dimensional parameter space of supersymmetric models, focusing computational resources on the most promising regions and improving the discovery of target regions.

