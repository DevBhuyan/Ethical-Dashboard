# The application is predominantly a web app, and can hence be accessed via a UI only
(Please let me know if you want it available as an API too)

## Here's a typical journey through the interface
This is the landing page where you'll land on opening the app

1. <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/6b26d9b9-d3e5-4ff8-a885-aa90e4706543" />

You can select a dataset from the available datasets in the dropdown, or you can upload your own custom CSV.
(Caveats: If you upload a custom dataset, ensure that the target column is named `Class`. Also, rows with missing attributes are dropped by default)

Once you have selected/uploaded your data, you'll reach the Dataset Summary screen

2. <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/c0262b11-06f8-4359-84a7-b17f2c1de076" />
3. <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/38b82972-c881-4255-ac76-1d01c76bce93" />

Where you can choose to **Edit** the data, or proceed to model selection/training. The "Edit" option will take you to a different page where you can manually change the values in the dataset (if a preselected dataset), or you can upload your own edited csv for convenience.

4. <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/bf5b462e-906d-4df7-8870-f4b27a0b2f8f" />

Once you're done with the dataset, you can proceed to choose/upload a model in the same way. (Caveats: If you upload a custom model, ensure it has a `fit(X_train, y_train, **args, **kwargs)` method. The deployed web-app has a file-size limit of 200MB, but if you run it locally, you should be able to load larger models as well, provided you have adequate compute)

5. <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/6d1a687d-02ce-48c9-b6d7-2d4671308b92" />

After model selection/uploading, you can preview its hyperparameters and start training. Once trained, you'll be able to view its training performance, viz. classification report, confustion matrix

6. <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/12b75967-7a2e-45b2-bba9-7f54264b9d4a" />

After this point, you will also see an ethical evaluation of the model-dataset, starting with:

## Fairness (evaluation of sensitive attributes against biases)

7. <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/af47b8b1-c4df-4e32-99a9-79987ca99301" />

## Robustness (we add Gaussian Noise to features and measure accuracy degradation)

8. <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/d8f0c042-15de-4529-9553-6524c4b8505c" />

## Privacy-Preserving (simulating a Membership-Inference-Attack)

9. <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/1ee87da1-df14-4ea6-ba7c-fe0a1a60f550" />


## Explainability (using SHAP)

10.<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/9806dbba-f8ce-4c36-a791-8ef41444cc8f" />
