import streamlit as st
import pandas as pd
import numpy as np
import xgboost
import cv2
import time
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
import os

# ---------------- Helper functions ----------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0: low = 0.001
    if high >= 1: high = 0.999
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

def compute_rppg_features(green_vals, red_vals, duration):
    """
    Returns dict with heart_rate, hrv(ms), resp_rate, spo2_est, bvp_amp
    Assumes green_vals & red_vals are lists of channel means.
    """
    out = {'heart_rate': None, 'hrv': None, 'resp_rate': None, 'spo2': None, 'bvp_amp': None}
    if len(green_vals) < 10:
        return out

    fs = len(green_vals) / duration
    green = (np.array(green_vals) - np.mean(green_vals)) / (np.std(green_vals) + 1e-6)
    red = (np.array(red_vals) - np.mean(red_vals)) / (np.std(red_vals) + 1e-6)

    # BVP from green channel
    try:
        bvp = butter_bandpass_filter(green, 0.7, 4.0, fs, order=3)
        # peaks for heart beats
        peaks, _ = find_peaks(bvp, distance=max( int(fs*0.4), 1 ))
        heart_rate = (len(peaks) / duration) * 60.0
        heart_rate = float(np.clip(heart_rate, 35, 200))
    except Exception:
        heart_rate = None
        peaks = []

    # HRV (ms) from inter-beat-intervals
    try:
        if len(peaks) > 1:
            ibi_sec = np.diff(peaks) / fs
            hrv_ms = float(np.std(ibi_sec) * 1000.0)
            hrv_ms = float(np.clip(hrv_ms, 5, 300))
        else:
            hrv_ms = 0.0
    except Exception:
        hrv_ms = 0.0

    # Respiratory rate from lower frequency band
    try:
        resp_signal = butter_bandpass_filter(green, 0.08, 0.6, fs, order=3)
        resp_peaks, _ = find_peaks(resp_signal, distance=max(int(fs*1.0),1))
        resp_rate = (len(resp_peaks) / duration) * 60.0
        resp_rate = float(np.clip(resp_rate, 6, 40))
    except Exception:
        resp_rate = None

    # Estimate SpO2: rough proxy using AC/DC of red/green ratio
    try:
        ac_red = np.std(red_vals)
        ac_green = np.std(green_vals)
        dc_red = np.mean(red_vals) + 1e-6
        dc_green = np.mean(green_vals) + 1e-6
        ratio = (ac_red/dc_red) / (ac_green/dc_green + 1e-6)
        # map ratio roughly to SpO2 range
        spo2_est = 100.0 - 20.0 * (ratio - 0.8)  # heuristic clamp
        spo2_est = float(np.clip(spo2_est, 85.0, 100.0))
    except Exception:
        spo2_est = None

    bvp_amp = float(np.clip(np.max(green) - np.min(green), 0.0, 5.0))

    out['heart_rate'] = round(heart_rate, 1) if heart_rate is not None else None
    out['hrv'] = round(hrv_ms, 1)
    out['resp_rate'] = round(resp_rate, 1) if resp_rate is not None else None
    out['spo2'] = round(spo2_est, 1) if spo2_est is not None else None
    out['bvp_amp'] = round(bvp_amp, 3)
    return out

def build_feature_row(inputs, vitals):
    """
    inputs: dict of manual clinical & lifestyle inputs (systolic, diastolic, cholesterol, hdl, etc.)
    vitals: dict from rPPG compute or manual entries (heart_rate, hrv, resp_rate, spo2, bvp_amp)
    Returns a single-row DataFrame with features matching your dataset.
    """
    row = {}
    # Demographic & lifestyle
    row['age'] = inputs.get('age', 40)
    row['gender'] = inputs.get('gender', 0)  # 0 female,1 male
    row['smoking'] = inputs.get('smoking', 0)
    row['alcohol_intake'] = inputs.get('alcohol_intake', 0)
    row['diet_score'] = inputs.get('diet_score', 5)
    row['physical_activity_level'] = inputs.get('physical_activity_level', 2)
    row['stress_level'] = inputs.get('stress_level', 5)
    row['sleep_hours'] = inputs.get('sleep_hours', 7)

    # Clinical
    row['systolic_bp'] = inputs.get('systolic_bp', 120)
    row['diastolic_bp'] = inputs.get('diastolic_bp', 80)
    row['cholesterol'] = inputs.get('cholesterol', 200)
    row['hdl_cholesterol'] = inputs.get('hdl_cholesterol', 50)
    row['ldl_cholesterol'] = inputs.get('ldl_cholesterol', 120)
    row['triglycerides'] = inputs.get('triglycerides', 150)
    row['blood_sugar_fasting'] = inputs.get('blood_sugar_fasting', 100)
    row['bmi'] = inputs.get('bmi', 24.0)
    row['family_history'] = inputs.get('family_history', 0)

    # rPPG vitals
    row['rppg_heart_rate'] = vitals.get('heart_rate', 70.0)
    row['rppg_hrv'] = vitals.get('hrv', 50.0)
    row['rppg_respiration_rate'] = vitals.get('resp_rate', 16.0)
    row['rppg_stress_index'] = vitals.get('rppg_stress_index', None)  # will compute below
    row['rppg_spo2_estimate'] = vitals.get('spo2', 98.0)
    row['rppg_blood_volume_pulse_amplitude'] = vitals.get('bvp_amp', 0.5)
    row['rppg_skin_temp'] = vitals.get('skin_temp', 34.0)

    # Derived
    row['mean_bp'] = (2.0 * row['diastolic_bp'] + row['systolic_bp']) / 3.0
    row['chol_hdl_ratio'] = row['cholesterol'] / (row['hdl_cholesterol'] + 1e-6)
    # bmi category
    if row['bmi'] < 25:
        row['bmi_category'] = 0
    elif row['bmi'] < 30:
        row['bmi_category'] = 1
    else:
        row['bmi_category'] = 2

    # rppg combined stress: same formula used for dataset creation
    # If user supplied rppg_stress_index, use it; else compute a proxy
    src_stress_index = vitals.get('rppg_stress_index', None)
    if src_stress_index is None:
        # Create a scaled stress index: higher HR, lower HRV increases stress
        hr_scaled = row['rppg_heart_rate'] / 120.0
        hrv_scaled = 1.0 - (row['rppg_hrv'] / 100.0)
        stress_proxy = 0.4 * 0.5 + 0.3 * hr_scaled + 0.3 * hrv_scaled  # baseline 0.5 if no value
        rppg_stress_idx = float(np.clip(stress_proxy * 100.0, 0.0, 100.0))
        row['rppg_stress_index'] = rppg_stress_idx
    else:
        row['rppg_stress_index'] = float(src_stress_index)

    row['rppg_combined_stress'] = (
        0.4 * (row['rppg_stress_index'] / 100.0)
        + 0.3 * (row['rppg_heart_rate'] / 120.0)
        + 0.3 * (1.0 - row['rppg_hrv'] / 100.0)
    )

    # Ensure numeric types
    for k, v in row.items():
        if isinstance(v, (np.floating, np.integer)):
            row[k] = float(v)
    return pd.DataFrame([row])

# ---------------- Load model ----------------
MODEL_PATH = 'xgb_model.bin'  # change to your actual model file path
loaded_model = None
try:
    loaded_model = xgboost.Booster()
    loaded_model.load_model(MODEL_PATH)
    st_model_loaded = True
except Exception as e:
    st_model_loaded = False
    loaded_model = None

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Heart Attack Prediction", page_icon="💓", layout="centered")
st.title('💓 Heart Attack Prediction using ML + rPPG Camera Analysis')

st.markdown("""<style>.main .block-container {max-width: 900px; padding-top: 2rem;}</style>""", unsafe_allow_html=True)

st.markdown(
    '<div style="display:flex;justify-content:center;"><img src="https://media.tenor.com/91scJf-xrKEAAAAi/emoji-coraz%C3%B3n-humano.gif" width="150"></div>',
    unsafe_allow_html=True,
)

# ---------------- Session State ----------------
if 'vitals' not in st.session_state:
    st.session_state['vitals'] = {
        'heart_rate': 70.0,
        'hrv': 50.0,
        'resp_rate': 16.0,
        'spo2': 98.0,
        'bvp_amp': 0.5,
        'skin_temp': 34.0,
        'rppg_stress_index': 40.0
    }

# ---------------- rPPG Capture ----------------
st.subheader("🎥 Measure Vitals using Camera (rPPG)")
st.write("Click 'Start Camera Measurement' to run a ~15s capture. Ensure even lighting and minimal movement.")

camera_col, manual_col = st.columns(2)

with camera_col:
    if st.button("Start Camera Measurement"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        green_vals, red_vals, blue_vals = [], [], []
        start_time = time.time()
        duration = 30
        st.info("📹 Recording... stay still and face the camera for 30 seconds.")

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Camera not detected.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb)

            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                face_landmarks = results.multi_face_landmarks[0]
                # forehead points - chosen heuristic indices from MediaPipe face mesh
                forehead_points = [10, 338, 297, 332, 284, 251]
                try:
                    pts = np.array([[int(face_landmarks.landmark[i].x * w),
                                     int(face_landmarks.landmark[i].y * h)] for i in forehead_points])
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)
                    g_mean = np.mean(frame[:, :, 1][mask == 255]) if np.any(mask==255) else np.mean(frame[:,:,1])
                    r_mean = np.mean(frame[:, :, 2][mask == 255]) if np.any(mask==255) else np.mean(frame[:,:,2])
                    b_mean = np.mean(frame[:, :, 0][mask == 255]) if np.any(mask==255) else np.mean(frame[:,:,0])
                except Exception:
                    g_mean = np.mean(frame[:, :, 1])
                    r_mean = np.mean(frame[:, :, 2])
                    b_mean = np.mean(frame[:, :, 0])

                green_vals.append(float(g_mean))
                red_vals.append(float(r_mean))
                blue_vals.append(float(b_mean))

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        cv2.destroyAllWindows()

        if len(green_vals) >= 10:
            vitals_result = compute_rppg_features(green_vals, red_vals, duration)
            # try to estimate a skin temp proxy from mean green intensity (very rough)
            vitals_result['skin_temp'] = round(33.0 + (np.mean(green_vals) - np.min(green_vals)) / 50.0, 1)
            # store rppg stress index also
            vitals_result['rppg_stress_index'] = 0.4 * (vitals_result['hrv'] and (100 - vitals_result['hrv']) or 50) + 0.3 * (vitals_result['heart_rate'] / 120.0) * 100
            # Save to session
            st.session_state['vitals'].update(vitals_result)
            st.success("✅ Recording completed & vitals estimated.")
            st.write(f"❤️ Heart Rate: {st.session_state['vitals']['heart_rate']} bpm")
            st.write(f"📉 HRV: {st.session_state['vitals']['hrv']} ms")
            st.write(f"🫁 Resp Rate: {st.session_state['vitals']['resp_rate']} bpm")
            st.write(f"🩸 SpO₂ (est): {st.session_state['vitals']['spo2']} %")
        else:
            st.error("❌ Could not capture usable rPPG signal. Try again with better lighting and steadier face.")

with manual_col:
    st.markdown("**Manual rPPG inputs (if camera fails)**")
    m_hr = st.number_input("Manual heart rate (bpm)", min_value=30.0, max_value=200.0, value=float(st.session_state['vitals']['heart_rate']))
    m_hrv = st.number_input("Manual HRV (ms)", min_value=1.0, max_value=500.0, value=float(st.session_state['vitals']['hrv']))
    m_resp = st.number_input("Manual respiration rate (bpm)", min_value=6.0, max_value=60.0, value=float(st.session_state['vitals']['resp_rate']))
    m_spo2 = st.number_input("Manual SpO₂ (%)", min_value=50.0, max_value=100.0, value=float(st.session_state['vitals']['spo2']))
    use_manual = st.checkbox("Use manual rPPG entries (override captured)", value=False)

    if use_manual:
        st.session_state['vitals'].update({
            'heart_rate': float(m_hr),
            'hrv': float(m_hrv),
            'resp_rate': float(m_resp),
            'spo2': float(m_spo2),
            'bvp_amp': float(st.session_state['vitals'].get('bvp_amp', 0.5)),
            'skin_temp': float(st.session_state['vitals'].get('skin_temp', 34.0))
        })
        st.info("✅ Manual rPPG values applied.")

# ---------------- Manual clinical inputs ----------------
st.subheader("🩺 Clinical & Lifestyle Inputs (fill all fields for best prediction)")
with st.form("clinical_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        gender = st.selectbox("Gender", ("Female", "Male"))
        smoking = st.selectbox("Smoking (0=No,1=Yes)", (0,1))
        alcohol_intake = st.selectbox("Alcohol (0=No,1=Yes)", (0,1))
        diet_score = st.slider("Diet score (1 poor - 10 excellent)", 1, 10, 5)
        physical_activity_level = st.slider("Physical activity (1 sedentary - 5 very active)", 1, 5, 2)
        stress_level = st.slider("Stress level (1 low - 10 high)", 1, 10, 5)

    with col2:
        sleep_hours = st.slider("Sleep hours", 0.0, 12.0, 7.0)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=220, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
        cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        hdl_cholesterol = st.number_input("HDL (mg/dL)", min_value=10, max_value=150, value=50)
        ldl_cholesterol = st.number_input("LDL (mg/dL)", min_value=10, max_value=300, value=120)
        triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=10, max_value=500, value=140)
        blood_sugar_fasting = st.number_input("Fasting blood sugar (mg/dL)", min_value=50, max_value=400, value=100)
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

        # Calculate BMI
        bmi = weight_kg / ((height_cm/100.0)**2)
        bmi = round(bmi, 1)

        st.markdown(f"**Calculated BMI:** {bmi}")
        family_history = st.selectbox("Family history of heart disease", (0,1))

    submitted = st.form_submit_button("Save Inputs")

    if submitted:
        st.success("✅ Clinical inputs saved to session.")
        st.session_state['clinical_inputs'] = {
            'age': int(age),
            'gender': 1 if gender == 'Male' else 0,
            'smoking': int(smoking),
            'alcohol_intake': int(alcohol_intake),
            'diet_score': int(diet_score),
            'physical_activity_level': int(physical_activity_level),
            'stress_level': int(stress_level),
            'sleep_hours': float(sleep_hours),
            'systolic_bp': float(systolic_bp),
            'diastolic_bp': float(diastolic_bp),
            'cholesterol': float(cholesterol),
            'hdl_cholesterol': float(hdl_cholesterol),
            'ldl_cholesterol': float(ldl_cholesterol),
            'triglycerides': float(triglycerides),
            'blood_sugar_fasting': float(blood_sugar_fasting),
            'bmi': float(bmi),
            'family_history': int(family_history)
        }

# ---------------- Prediction ----------------
st.markdown("---")
st.header("🔍 Predict Heart Attack Risk")

if not st_model_loaded:
    st.warning("Model not loaded. Please ensure xgb_model.bin exists in the app directory.")
else:
    if st.button("Predict Risk"):
        clinical = st.session_state.get('clinical_inputs', None)
        if clinical is None:
            st.error("Please fill and submit clinical & lifestyle inputs first.")
        else:
            vitals = st.session_state['vitals']
            feature_df = build_feature_row(clinical, vitals)

            st.write("### 📋 Model Inputs")
            st.dataframe(feature_df.T, width=800)

            # Ensure feature order matches training order
            # Provide the list in the same order used while training the model
            feature_cols = [
                'age','gender','smoking','alcohol_intake','diet_score','physical_activity_level','stress_level','sleep_hours',
                'systolic_bp','diastolic_bp','cholesterol','hdl_cholesterol','ldl_cholesterol','triglycerides',
                'blood_sugar_fasting','bmi','family_history',
                'rppg_heart_rate','rppg_hrv','rppg_respiration_rate','rppg_stress_index','rppg_spo2_estimate',
                'rppg_blood_volume_pulse_amplitude','rppg_skin_temp',
                'mean_bp','chol_hdl_ratio','bmi_category','rppg_combined_stress'
            ]

            # Some models expect exactly these columns. Fill missing with zeros.
            for c in feature_cols:
                if c not in feature_df.columns:
                    feature_df[c] = 0.0

            feature_df = feature_df[feature_cols].astype(float)

            # Create DMatrix
            dtest = xgboost.DMatrix(feature_df.values, feature_names=feature_cols)

            # Predict
            try:
                pred = loaded_model.predict(dtest)
                # XGBoost Booster.predict can return:
                # - single column for regression / binary raw score
                # - n x k array for multiclass probabilities
                if pred.ndim == 2 and pred.shape[1] > 1:
                    # multiclass probabilities
                    class_idx = int(np.argmax(pred, axis=1)[0])
                    prob = pred[0, class_idx]
                    label_map = {0: "Low", 1: "Medium", 2: "High"}
                    risk_label = label_map.get(class_idx, str(class_idx))
                    st.success(f"Risk: {risk_label}  (probability={prob:.2f})")
                else:
                    # single score -> map using threshold(s)
                    score = float(pred[0])  # raw margin or prob depending on training
                    # If model outputs logit-like values, convert sigmoid
                    if score < -1 or score > 1:
                        # try to convert using sigmoid
                        prob = 1.0 / (1.0 + np.exp(-score))
                    else:
                        prob = score if 0 <= score <= 1 else 1.0 / (1.0 + np.exp(-score))
                    # map to 3 classes using thresholds (tune these after validation)
                    if prob < 0.40:
                        risk_label = "Low"
                    elif prob < 0.55:
                        risk_label = "Medium"
                    else:
                        risk_label = "High"
                    st.success(f"Risk: {risk_label}  (probability={prob:.2f})")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

            # Optional: save the input row for debugging / later analysis
            log_dir = "prediction_logs"
            os.makedirs(log_dir, exist_ok=True)
            fname = os.path.join(log_dir, f"input_{int(time.time())}.csv")
            feature_df.to_csv(fname, index=False)
            st.info(f"Input saved for debugging: {fname}")

st.markdown("---")
st.caption("Developed by Renuka 💻 | Use responsibly — this is NOT a clinical diagnosis. Always consult qualified medical professionals for medical advice.")
