import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="نظام تقييم مخاطر أمراض القلب", page_icon="❤️", layout="centered"
)
st.markdown(
    """
    <style>
        html, body, [class*="css"], div[data-testid="stForm"], div[data-baseweb="select"] {
        direction: rtl !important;
        text-align: right !important;
    }
    .stSelectbox, .stNumberInput {
        direction: rtl !important;
        text-align: right !important;
    }
    .main-title {
        color: #b30000;
        text-align: center;
        font-weight: bold;
        margin-bottom: 25px;
    }
    div.stButton > button:first-child {
        background-color: #d9534f !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #c9302c !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 class='main-title'> نظام تقييم مخاطر أمراض القلب</h1>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
  model = joblib.load("heart_model.pkl")
  scaler = joblib.load("scaler.pkl")
  return model, scaler


try:
  model, scaler = load_assets()
except Exception:
  st.error(
      "يرجى تشغيل ملف التدريب train.py أولاً لإنشاء ملفات heart_model.pkl و"
      " scaler.pkl الحديثة"
  )
  st.stop()


def reset_form():
  st.session_state["age"] = 50
  st.session_state["sex"] = 1
  st.session_state["cp"] = 2
  st.session_state["restbp"] = 120
  st.session_state["chol"] = 200
  st.session_state["fbs"] = 0
  st.session_state["restecg"] = 0
  st.session_state["maxhr"] = 150
  st.session_state["exang"] = 0
  st.session_state["oldpeak"] = 1.0
  st.session_state["slope"] = 0


tab1, tab2 = st.tabs([" البيانات الأساسية", " الفحوصات الطبية"])

with tab1:
  col1, col2 = st.columns(2)
  with col1:
    age = st.number_input("العمر (بالسنوات)", 1, 120, key="age", value=50)
    sex = st.selectbox(
        "الجنس",
        options=[1, 0],
        format_func=lambda x: "ذكر" if x == 1 else "أنثى",
        key="sex",
    )
    fbs = st.selectbox(
        "فحص السكر الصائم (أعلى من 120؟)",
        [1, 0],
        format_func=lambda x: "نعم" if x == 1 else "لا",
        key="fbs",
    )

  with col2:
    restbp = st.number_input(
        "ضغط الدم في حالة الراحة (Resting BP)", 80, 220, key="restbp", value=120
    )
    chol = st.number_input(
        "مستوى الكوليسترول في الدم (Cholesterol)",
        0,
        600,
        key="chol",
        value=200,
    )

with tab2:
  col1, col2 = st.columns(2)
  with col1:
    cp_opts = {
        0: "ألم غير نموذجي (ATA)",
        1: "ألم بدون ذبحة صدرية (NAP)",
        2: "بدون أعراض / صامت (ASY)",
        3: "ألم الذبحة الصدرية النموذجي (TA)",
    }
    cp = st.selectbox(
        "طبيعة ألم الصدر (Chest Pain Type)",
        list(cp_opts.keys()),
        format_func=lambda x: cp_opts[x],
        key="cp",
    )

    ecg_opts = {
        0: "طبيعي (Normal)",
        1: "اضطراب الموجة (ST)",
        2: "تضخم عضلة القلب (LVH)",
    }
    restecg = st.selectbox(
        "نتيجة تخطيط القلب (Resting ECG)",
        list(ecg_opts.keys()),
        format_func=lambda x: ecg_opts[x],
        key="restecg",
    )

    maxhr = st.number_input(
        "أعلى معدل لنبض القلب عند المجهود (Max HR)",
        60,
        220,
        key="maxhr",
        value=150,
    )

  with col2:
    exang = st.selectbox(
        "هل يظهر ألم الصدر أثناء ممارسة الرياضة؟ (Exercise Angina)",
        [1, 0],
        format_func=lambda x: "نعم" if x == 1 else "لا",
        key="exang",
    )

    oldpeak = st.number_input(
        "مستوى إجهاد القلب (Oldpeak)",
        -2.0,
        10.0,
        key="oldpeak",
        value=1.0,
        step=0.1,
    )

    slope_opts = {
        0: "صاعد / تصاعدي (Up)",
        1: "مستوي / أفقي (Flat)",
        2: "هابط / انحداري (Down)",
    }
    slope = st.selectbox(
        "استجابة نبض القلب للمجهود (ST Slope)",
        list(slope_opts.keys()),
        format_func=lambda x: slope_opts[x],
        key="slope",
    )

st.markdown("---")
btn_col1, btn_col2 = st.columns([3, 1])

with btn_col1:
  evaluate = st.button(" تقييم الحالة", use_container_width=True)

with btn_col2:
  st.button(" إعادة تعيين", on_click=reset_form, use_container_width=True)

if evaluate:
  input_data = np.array([[
      age,
      sex,
      cp,
      restbp,
      chol,
      fbs,
      restecg,
      maxhr,
      exang,
      oldpeak,
      slope,
  ]])

  input_data_scaled = scaler.transform(input_data)

  prediction = model.predict(input_data_scaled)[0]
  prob = model.predict_proba(input_data_scaled)[0][1]

  st.subheader("مؤشر نتيجة التقييم الطبي:")
  st.progress(float(prob))

  if prediction == 1:
    st.error(
        f"⚠️ هناك احتمالية عالية للإصابة بأمراض القلب بنسبة ({prob * 100:.1f}%)."
    )
  else:
    st.success(
        f"✅ الحالة سليمة بنسبة عالية (احتمالية الخطر {prob * 100:.1f}% فقط)."
    )