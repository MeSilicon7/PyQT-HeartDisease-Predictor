import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QPushButton, QMessageBox

# Load the dataset and prepare the model
heart_data = pd.read_csv('heart.csv')
label_encoders = {}
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in categorical_columns:
    le = LabelEncoder()
    heart_data[col] = le.fit_transform(heart_data[col])
    label_encoders[col] = le

X = heart_data.drop('HeartDisease', axis=1)
y = heart_data['HeartDisease']
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

class HeartDiseasePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.labels = {}
        self.entries = {}
        layout = QVBoxLayout()

        for col in X.columns:
            self.labels[col] = QLabel(f'Enter {col}')
            self.entries[col] = QLineEdit(self)
            layout.addWidget(self.labels[col])
            layout.addWidget(self.entries[col])

        self.predict_btn = QPushButton('Predict', self)
        self.predict_btn.clicked.connect(self.on_predict)
        layout.addWidget(self.predict_btn)

        self.setLayout(layout)
        self.setWindowTitle('Heart Disease Predictor')
        self.show()

    def on_predict(self):
        patient_data = {}

        for col in X.columns:
            value = self.entries[col].text()
            if col in categorical_columns:
                value = label_encoders[col].transform([value])[0]
            else:
                value = float(value)
            patient_data[col] = [value]

        patient_df = pd.DataFrame(patient_data)
        prediction = clf.predict(patient_df)

        if prediction[0] == 1:
            QMessageBox.information(self, 'Prediction', 'The patient is likely suffering from heart disease.')
        else:
            QMessageBox.information(self, 'Prediction', 'The patient is likely not suffering from heart disease.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HeartDiseasePredictor()
    sys.exit(app.exec_())
