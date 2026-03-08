# Ethical Dashboard

An interactive dashboard for evaluating **Responsible AI properties** of machine learning models.

The Ethical Dashboard allows researchers and developers to test machine learning models across four critical ethical dimensions:

- **Fairness**
- **Robustness**
- **Privacy Preservation**
- **Explainability**

The tool provides an end-to-end interface to upload datasets, train or upload models, and analyze the ethical implications of model predictions.

A live demo of the dashboard is available here:

https://ethical-dashboard.streamlit.app/

You can also use [how_to_use_the_app.md](https://github.com/DevBhuyan/Ethical-Dashboard/blob/main/how_to_use_the_app.md) as an interactive guide for "how-to-use".

---

# Features

### Dataset Inspection
- Upload custom datasets (CSV format)
- Use built-in benchmark datasets
- Visualize dataset distributions and statistics
- Edit datasets directly inside the dashboard

### Model Integration
Users can:

- Select predefined models
- Upload their own trained models

Supported model formats include:
1. `.pkl`
2. `.h5`
3. `.hdf5`
4. `.safetensors`


Supported frameworks:

- scikit-learn
- TensorFlow / Keras
- PyTorch

The dashboard automatically inspects models and displays:

- model type
- hyperparameters
- architecture summary (when available)

---

# Responsible AI Evaluation Modules

After training or loading a model, the dashboard evaluates it using four Responsible AI modules.

## 1. Fairness

Evaluates prediction disparities across sensitive groups.

Metrics include:

- Accuracy per group
- Precision
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Selection rate
- Group population statistics

This helps detect potential **bias in model predictions**.

---

## 2. Robustness

Measures how model performance changes under input perturbations.

The dashboard evaluates robustness by applying:

- Gaussian noise to inputs
- Perturbation-based stress testing

Outputs include:

- accuracy degradation curves
- robustness visualizations

This helps detect models that are **overly sensitive to small changes in input data**.

---

## 3. Privacy Preservation

Evaluates privacy risks using **Membership Inference Attacks (MIA)**.

The module estimates whether an attacker can infer whether a sample was used in the training dataset.

Metrics include:

- attack accuracy
- attack precision
- attack recall
- attack F1-score
- attack confusion matrix

This helps estimate **training data leakage risk**.

---

## 4. Explainability

Uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

The module provides:

- global feature importance
- local explanations for individual predictions
- explanation stability metrics
- explanation sparsity metrics

This helps interpret **which features influence model predictions**.

---

# Installation

### Clone the repository:

```
git clone https://github.com/DevBhuyan/Ethical-Dashboard.git
cd Ethical-Dashboard
```

### Install dependencies
`python -m pip install -r requirements.txt`
Or if you prefer `uv`
`uv pip install -r requirements.txt`

# Running the Dashboard locally

`streamlit run app.py`

Then open a browser at `http://localhost:8501`

You must see the landing page as such:
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/a1702641-6278-4614-94b0-161b3d0975b7" />

Happy evaluating!

For more information on how to use the app, see [how_to_use_the_app.md](https://github.com/DevBhuyan/Ethical-Dashboard/blob/main/how_to_use_the_app.md)
