# ❤️ Heart Disease Risk Prediction System

A Machine Learning-based web application designed to predict and evaluate heart disease risk using clinical parameters, built with Python and Streamlit.

---

### 🚀 Features
* **Risk Assessment:** Analyzes health parameters using a trained Machine Learning model (`heart_model.pkl`).
* **Data Processing:** Utilizes `scaler.pkl` to preprocess and scale user input data accurately.
* **Interactive Web UI:** Built with **Streamlit** for a smooth, fast, and user-friendly interface.
* **Model Training Script:** Includes `train.py` to demonstrate how the model was trained on the dataset (`Heart.csv`).

---

### 🛠️ Tech Stack
* **Python**
* **Streamlit** (Web Framework)
* **Scikit-Learn / Pandas** (Machine Learning & Data Processing)

---

### ⚙️ How to Setup & Run

1. **Open the Project in VS Code:**
   * Open **Visual Studio Code**, select **File > Open Folder**, and choose this project's folder.
   * Open the integrated terminal (`Ctrl + ~`).

2. **Install Dependencies:**
   Run the following command in your terminal to install the required libraries:
   ```bash
   pip install streamlit pandas scikit-learn
   
3. **Train the Model (Optional):**
If you want to train the model using the dataset (Heart.csv), run:
   ```bash
   python train.py
 Or using your local Python path:
       ```bash
       %LocalAppData%\Programs\Python\Python314\python.exe train.py

4. **Run the Application:**
 Execute the following command in your terminal to launch the web app:
   ```bash
 streamlit run app.py
