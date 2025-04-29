# Spam-Email-Detection-using-Naïve-Bayes

## College Project

Welcome to the **Spam Email Detection using Naïve Bayes** repository! This project is part of a college assignment to classify emails as spam or non-spam using the Naïve Bayes algorithm in Python.

---

## Project Overview

Spam detection is a vital aspect of email management, ensuring users receive only genuine messages. The goal of this project is to implement the **Naïve Bayes algorithm** for effective spam classification.

### **Problem Statement:**
Given a collection of emails, classify each email as **spam** or **non-spam** (ham). Spam emails often contain unwanted advertising, phishing attempts, or irrelevant information, while ham emails are legitimate messages.

---

## Dataset

The dataset used for this project is taken from **Kaggle Spam Detection** and contains over **5000 emails**, categorized as either **spam** or **ham**. Each email contains a label (`spam` or `ham`) and the corresponding email content.

- **Dataset Source:** Kaggle Spam Detection
- **Columns:**
  - `label`: The label for each email (`spam` or `ham`).
  - `message`: The content of the email.

---

## Project Workflow

### **1. Data Preprocessing**
   - **Text Cleaning**: 
     - Remove non-alphabetic characters and convert text to lowercase.
     - Remove unnecessary white spaces.
   - **Label Encoding**: 
     - Convert labels into numeric format: `spam` = 1, `ham` = 0.

### **2. Feature Extraction**
   - **TF-IDF Vectorization**: 
     - Convert the email text to numeric feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)** to represent word importance.

### **3. Model Building**
   - **Naïve Bayes Classifier**: 
     - The **Multinomial Naïve Bayes** algorithm is used to classify the emails based on the word frequencies in each email.

### **4. Model Evaluation**
   - **Accuracy Score**: 
     - Measure the performance of the model by calculating the accuracy.
   - **Classification Report**: 
     - Includes **precision**, **recall**, and **F1-score** to evaluate the model’s classification performance.

---

## Libraries & Tools Used

- **Python**: Programming language.
- **Pandas**: For data manipulation and cleaning.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning (Naïve Bayes classifier, TF-IDF, evaluation metrics).
- **Matplotlib/Seaborn**: For data visualization (optional for extra analysis).

---

## Installation & Setup

### **Requirements**
Ensure you have Python installed (preferably Python 3.6+). You can set up a virtual environment or directly install the following dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### **Running the Code**

1. **Download or Clone the Repository**:
   - You can either **clone** this repository using Git or **download** it as a ZIP file.
   
   ```bash
   git clone https://github.com/your-username/Spam-Email-Detection-using-Naive-Bayes.git
   ```

2. **Upload the Dataset**:
   - Make sure the dataset (`spam.csv`) is available in the working directory or linked from Google Drive if using Google Colab.

3. **Execute the Python Script**:
   - You can run the script using any IDE or Jupyter Notebook/Google Colab.
   - Alternatively, you can run the entire notebook in Google Colab directly.

---

## Example Output

After running the code, you will see the accuracy and the detailed classification report as output.

**Sample Output:**

```
Accuracy: 97.89%

Classification Report:
              precision    recall  f1-score   support

         Ham       0.99      0.99      0.99       144
        Spam       0.93      0.92      0.93        24

    accuracy                           0.98       168
   macro avg       0.96      0.96      0.96       168
weighted avg       0.98      0.98      0.98       168
```

---

## Future Improvements

- **Hyperparameter Tuning**: Tune the parameters of the Naïve Bayes model to improve performance.
- **Advanced Models**: Use advanced techniques like **Deep Learning (RNN, LSTM, or BERT)** for improved classification.
- **Real-time Prediction**: Create a web app (using **Flask** or **Streamlit**) where users can input their email text for real-time spam detection.

---

## Author

**Name**: Satyam Thakur 
**College/University**:LLoyd institute of technology and engineering
**Date**: April 2025

---

## License

This project is open-source and available under the **MIT License**.

---
🤝 Connect With Us
We’d love to hear from you! Feel free to connect with us for feedback, suggestions, or collaboration opportunities.

Follow Us on LinkedIn:

🔗 www.linkedin.com/in/satyam-thakur-674ba9330
