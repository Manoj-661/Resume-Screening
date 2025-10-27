🧾 Resume Category Classification using Machine Learning

This project automatically classifies resumes into job categories using Natural Language Processing (NLP) and Machine Learning techniques. A Streamlit web app is deployed for real-time classification from uploaded PDF/DOCX resumes.

🚀 Project Overview

| Feature            | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| **Goal**           | Predict the job role/category from resume text                  |
| **Dataset**        | Kaggle Resume Dataset                                           |
| **Classes**        | 25 Job Categories (e.g., Data Science, Web Designing, HR, etc.) |
| **Final Model**    | Support Vector Classifier (SVC)                                 |
| **Framework Used** | Streamlit                                                       |
| **Language**       | Python                                                          |

📊 Job Categories Covered

Data Science

Web Designing

HR

Python Developer

DevOps Engineer

Java Developer

Mechanical Engineer

Sales

Blockchain

Business Analyst

Testing

Health and Fitness

And more… (total 25 categories ✅)

🔬 Machine Learning Models Used
Model	Accuracy
K-Nearest Neighbors (KNN)	✅ Implemented
SVC (Support Vector Classifier)	✅ Final / Best Performance
Random Forest Classifier	✅ Tried

✅ SVC was selected as the final model due to best performance & stability.

🧠 NLP Techniques Used

Text cleaning & preprocessing

Tokenization

Lemmatization

TF-IDF vectorization

Stopword removal


📌 Web App Preview

Upload a resume → Extract text → Predict job category ✅

🛠️ Libraries Used
| Library        | Purpose                       |
| -------------- | ----------------------------- |
| Scikit-learn   | Machine Learning              |
| NLTK           | NLP Processing                |
| pdfplumber     | Extract text from PDF resumes |
| docx2txt       | Extract text from DOCX        |
| Streamlit      | Web deployment                |
| NumPy / Pandas | Data handling                 |


✅ Conclusion

This project demonstrates a full ML deployment pipeline:

✅ Dataset Preprocessing
✅ Multi-class Classification
✅ Model Saving
✅ Interactive Web Interface

🚀 Successfully transforms a resume into a predicted job role — useful for HR automation & job portals!
