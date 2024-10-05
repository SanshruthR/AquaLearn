# AquaLearn: Automated Machine Learning 
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.23.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![H2O](https://img.shields.io/badge/H2O-3.40.0-00FFFF?style=for-the-badge&logo=h2o&logoColor=black)
[![Deployed on Hugging Face](https://img.shields.io/badge/Deployed%20on-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Sanshruth/AquaLearn)
![image](https://github.com/user-attachments/assets/4443073f-5bf6-448c-bc6f-3b84300bd1aa)


## Overview
AquaLearn is a powerful and user-friendly AutoML application built with Streamlit and H2O.ai. It allows users to easily upload datasets, train machine learning models, and make predictions, all through an intuitive web interface.

### Features
- **CSV File Upload**: Easy data import with preview functionality
- **Automatic Feature Preparation**: Streamlined data preprocessing
- **AutoML**: Customizable algorithms and parameters for optimal model selection
- **Model Performance Visualization**: Clear insights into model performance
- **Model Saving and Downloading**: Preserve and share your trained models
- **Further Model Training**: Refine models with additional training
- **Easy Predictions**: Make predictions using uploaded models

### How It Works
1. Upload your CSV dataset
2. Select problem type (Classification or Regression)
3. Choose target column and algorithms
4. Run AutoML with customizable parameters
5. View results, save models, and make predictions

### Usage
Visit the [AquaLearn Hugging Face Space](https://huggingface.co/spaces/Sanshruth/AquaLearn) to start using the application immediately.

### Local Development
To run AquaLearn locally:
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

### Docker
To run AquaLearn using Docker:
1. Build the Docker image: `docker build -t aqualearner .`
2. Run the container: `docker run -p 7860:7860 aqualearner`

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### License
This project is licensed under the MIT License.
