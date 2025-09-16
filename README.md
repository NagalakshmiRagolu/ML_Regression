📌 Regression Project

This project is based on Machine Learning Regression Techniques. It focuses on predicting outcomes using both Simple Linear Regression (SLR) and Multiple Linear Regression (MLR) methods.

🔹 Project Overview
Simple Linear Regression (SLR):

Used to predict the salary of an employee based on their years of experience.

Multiple Linear Regression (MLR):

Used to predict the profit of startups based on various factors such as R&D spend, administration cost, marketing spend, and state.

The project is divided into two parts:

Backend (Model Training & Prediction): Machine Learning models are trained using datasets and saved for further use.

Frontend (User Interaction): Provides options to choose between SLR and MLR, input values, and view predicted outputs.

🔹 Methods Used

Regression Techniques

Simple Linear Regression (SLR)

Multiple Linear Regression (MLR)

Evaluation Metrics

R² Score (Accuracy)

Mean Squared Error (Loss)

Visualization

Matplotlib is used to plot the training and testing results for better understanding of predictions.

🔹 Technologies & Modules Used

Programming Language: Python

Libraries & Frameworks:

NumPy

Pandas

Matplotlib

Scikit-learn

Pickle (for saving models)

Flask (for frontend integration)

🔹 Project Flow
1. Data Collection

Salary Data for SLR

Startup Data for MLR

2. Data Preprocessing

Splitting datasets into training and testing sets

Encoding categorical values (for MLR)

3. Model Training

Train SLR and MLR models using Scikit-learn

Evaluate models with accuracy and error metrics

4. Visualization

Graphs to compare actual vs predicted values

5. Model Saving

Models are stored in .pkl files for future predictions

6. Frontend Integration

Home page → Access SLR & MLR predictions

About page → Explains the project workflow

Contact page → General project details
