# deep-learning-challenge
## 📊 Alphabet Soup Charity: Neural Network Analysis Report
📌 Overview of the Analysis
The purpose of this analysis was to build a deep learning model that could predict whether applicants to Alphabet Soup’s funding program would be successful. By training a neural network on historical data—including organization type, income category, use case, and application classification—we aimed to automate and improve the charity’s decision-making process.

Using TensorFlow and Keras, multiple versions of a binary classification model were created and evaluated for predictive performance.

## 📈 Results
### 🔹 Data Preprocessing
Target Variable:

IS_SUCCESSFUL: a binary variable (1 = successful application, 0 = unsuccessful).

Feature Variables:

All other columns in the dataset were treated as features after preprocessing.

Categorical columns such as APPLICATION_TYPE, CLASSIFICATION, AFFILIATION, USE_CASE, and INCOME_AMT were encoded using pd.get_dummies().

Dropped Columns:

EIN and NAME: dropped as non-predictive identifiers.

Additional low-signal features (USE_CASE_CommunityServ, INCOME_AMT_25000-99999, and AFFILIATION_Independent) were dropped in optimization attempt #4 to reduce noise.

Other Preprocessing Actions:

Rare values in APPLICATION_TYPE and CLASSIFICATION were grouped into an "Other" category to prevent high dimensionality and class sparsity.

All features were scaled using StandardScaler.

### 🔹 Model Architectures, Training, and Evaluation
✅ Baseline Model (nn)
Layers: 2 hidden layers with 80 and 30 neurons

Activations: ReLU

Epochs: 100

Accuracy: ~72.95%

## 🔁 Optimization Attempts
Model	Key Changes	Accuracy
nn_opt_1	2 layers, 100 & 50 neurons, 20 epochs	~72.98%
nn_opt_2	3 layers (128-64-32), tanh activation, 30 epochs	73.08%
nn_opt_3	3 layers (100-50-25), relu + Dropout(0.2), 30 epochs	~72.92%
nn_opt_4	Dropped noisy features, 3 layers (100-64-32), EarlyStopping, 100 epochs	~72.76%

### ✅ Optimization Techniques Attempted
Varying the number of layers and neurons.

Testing different activation functions: relu and tanh.

Adding Dropout layers to prevent overfitting.

Using EarlyStopping with validation split.

Dropping potentially noisy features based on domain judgment.

## 🧠 Summary and Recommendation
The highest accuracy achieved was 73.08%, falling just short of the target threshold of 75%. However, the model demonstrated consistent and reliable performance across multiple architectures and optimization strategies.

## 📝 Final Recommendation
If interpretability and slight performance improvements are desired, transitioning to tree-based ensemble models could help surpass the 75% target while offering feature importance metrics. However, the best-performing neural network (nn_opt_2) is still a solid candidate for automating application triage at Alphabet Soup.

## Acknowledgments
Model was prepared using the starter file and with the assistance of prior class examples and Xpert Learning Assistant. 