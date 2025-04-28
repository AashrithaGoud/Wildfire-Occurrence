import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"


def preprocess_data(fire_df):
    le = LabelEncoder()
    fire_df["ASPECT"] = le.fit_transform(fire_df["ASPECT"])
    fire_df["STATISTICAL_CAUSE"] = le.fit_transform(fire_df["STATISTICAL_CAUSE"])
    fire_df["STATE_NAME"] = le.fit_transform(fire_df["STATE_NAME"])

    X = fire_df[["STATE_NAME", "LATITUDE", "LONGITUDE","SLOPE", "ASPECT", "ELEVATION"]]
    y = fire_df["FIRE_INTENSITY_LEVEL"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=0, stratify=y)

def random_forest_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10,
                                   min_samples_leaf=4, max_features='sqrt', random_state=0, n_jobs = -1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Random Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def fcnn_model(X_train, X_test, y_train, y_test):
    num_classes = len(np.unique(y_train))
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)

    print("FCNN Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
