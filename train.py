import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Heart.csv")
#الجنس
df["Sex"] = df["Sex"].astype(str).str.strip().map({"M": 1, "F": 0})

#نوع الم الصدر
# TA: ذبحة نمطية (ألم قلب اثناء المجهود أو الرياضة)
# ATA: ألم غير نمطي ( ألم قلب اثناء الراحة أو عند التوتر)
# NAP: ألم غير ذبحي (عضلي أو بسبب المعدة ليس له علاقة بالقلب)
# ASY: بدون ألم (عرض صامت)
chestpain_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
df["ChestPainType"] = df["ChestPainType"].astype(str).str.strip().map(chestpain_map)

#تخطيط القلب 
# Normal: رسم القلب طبيعي وسليم 100%
# ST: اضطراب مؤقت في الموجة (إجهاد بعضلة القلب)
# LVH: تضخم في عضلة الجانب الأيسر للقلب
ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
df["RestingECG"] = df["RestingECG"].astype(str).str.strip().map(ecg_map)

#الم المجهود
# Y: نعم (يشعر بألم في الصدر عند ممارسة الرياضة أو المجهود)
# N: لا (لا يشعر بأي ألم في الصدر عند المجهود)
df["ExerciseAngina"] = df["ExerciseAngina"].astype(str).str.strip().map({"Y": 1, "N": 0})

#مخطط STميل قطاع 
# Up: صاعد (استجابة طبيعية وسليمة)
# Flat: مستوٍ (استجابة ضعيفة وتدل على إجهاد)
# Down: هابط (مؤشر قوي على مشكلة بالشرايين)
slope_map = {"Up": 0, "Flat": 1, "Down": 2}
df["ST_Slope"] = df["ST_Slope"].astype(str).str.strip().map(slope_map)

feature_columns = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]

X = df[feature_columns]
y = df["HeartDisease"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, criterion="entropy"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy: {round(acc * 100, 2)}%\n")

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(
    classification_report(
        y_test, y_pred, target_names=["No Disease (0)", "Disease (1)"]
    )
)
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel & Scaler saved successfully for 918 dataset!")