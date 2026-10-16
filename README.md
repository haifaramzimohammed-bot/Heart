# ❤️ Heart Disease Risk Prediction System

A Machine Learning-based web application designed to predict and evaluate heart disease risk using clinical parameters, built with Python and Streamlit.

---

### 🚀 Features
* **Risk Assessment:** Analyzes health parameters using a trained Machine Learning model (`heart_model.pkl`).
* **Data Processing:** Utilizes `scaler.pkl` to preprocess and scale user input data accurately.
* **Interactive Web UI:** Built with **Streamlit** for a smooth, fast, and user-friendly interface.
* **Model Training Script:** Includes `train.py` to demonstrate how the model was trained on the dataset (`Heart.csv`).
* **Run Instructions:** Includes `how_run.txt` containing direct commands for training and launching the app.

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
   Run this command in your terminal: `pip install streamlit pandas scikit-learn`

3. **Train the Model (Optional):**
   Run this command: `python train.py`
   *(Or using your local path: `%LocalAppData%\Programs\Python\Python314\python.exe train.py`)*

4. **Run the Application:**
   Run this command to launch the web app: `streamlit run app.py`
