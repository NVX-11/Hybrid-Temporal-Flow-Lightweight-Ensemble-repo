# -----------------------------------------------------------------
# STAGE 1: Imports & Environment Setup
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# -----------------------------------------------------------------
# STAGE 2: Preprocessing & Temporal Layer (EWMA)
# -----------------------------------------------------------------

def load_data():
    np.random.seed(42)
    normal = np.random.normal(loc=0.5, scale=0.1, size=(800, 10))
    attacks = np.random.normal(loc=1.2, scale=0.3, size=(200, 10))
    X = np.vstack([normal, attacks])
    y = np.array([0]*800 + [1]*200)
    return pd.DataFrame(X), y

df, labels = load_data()

#  تطبيق التنعيم الزمني (EWMA)
alpha = 0.3
df_smoothed = df.ewm(alpha=alpha).mean()

#  تقييس البيانات (Scaling)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_smoothed)


# -----------------------------------------------------------------
# STAGE 3: Model Definitions (Ensemble Components)
# -----------------------------------------------------------------

# بناء مشفر تلقائي خفيف الوزن
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(4, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# التدريب
history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=16, verbose=0)

#  حساب خطأ إعادة البناء من AE
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
dynamic_threshold = np.percentile(mse, 85) # عتبة ديناميكية
ae_preds = (mse > dynamic_threshold).astype(int)

#  تشغيل Isolation Forest
iso_forest = IsolationForest(contamination=0.2, random_state=42)
iso_preds = iso_forest.fit_predict(X_scaled)
iso_preds_binary = np.where(iso_preds == 1, 0, 1)

# 3. الدمج التجميعي (Ensemble Integration)
# إذا اكتشف أي نموذج شذوذا يتم اعتباره هجوماً
final_preds = np.logical_or(ae_preds, iso_preds_binary).astype(int)

# -----------------------------------------------------------------
# STAGE 4: Evaluation
# -----------------------------------------------------------------


print(classification_report(labels, final_preds))

# (Confusion Matrix)
cm = confusion_matrix(labels, final_preds)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: HTF-LE Model')
plt.show()


