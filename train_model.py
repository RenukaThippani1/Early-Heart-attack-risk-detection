import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load your dataset
df = pd.read_csv("early_heart_attack_prediction_1000.csv")

# 2. Split data into input features (X) and target (y)
X = df.drop("heart_attack_risk", axis=1)
y = df["heart_attack_risk"]

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Handle class imbalance
num_pos = sum(y_train == 1)
num_neg = sum(y_train != 1)
scale_weight = num_neg / (num_pos + 1e-6)

# 6. Train the XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(y.unique()),
    eval_metric='mlogloss',
    use_label_encoder=False,
    scale_pos_weight=scale_weight,
    random_state=42
)
model.fit(X_train, y_train)

# 7. Evaluate model on test set
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Save the trained model and scaler
model.save_model("xgb_model.bin")
joblib.dump(scaler, "scaler.bin")
print("✅ Model and scaler saved successfully as xgb_model.bin and scaler.bin")
