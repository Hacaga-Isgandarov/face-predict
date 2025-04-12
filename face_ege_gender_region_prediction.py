import cv2
from deepface import DeepFace
import numpy as np

# Race -> Country mapping
race_to_country = {
    'asian': 'Asiya',
    'white': 'Avropa',
    'black': 'Afrika',
    'middle eastern': 'Yaxin Sherq',
    'latino hispanic': 'Latin Amerikasi',
    'indian': 'Hindistan',
    'southeast asian': 'Cenub-Sherqi Asiya'
}

# Kamera aç
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera acila bilmədi.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame oxuna bilmədi.")
        break

    try:
        results = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)

        # Bir neçə üz tanınıbsa
        if isinstance(results, list):
            analysis_list = results
        else:
            analysis_list = [results]

        for result in analysis_list:
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
            age = int(result['age'])

            # Cinsiyyət (gender) düzəldilir
            gender_data = result['gender']
            if isinstance(gender_data, dict):
                gender_raw = max(gender_data, key=gender_data.get).lower()
            else:
                gender_raw = gender_data.lower()
            gender = 'Kisi' if gender_raw == 'man' else 'Qadin'

            # Milliyyət (race) düzəldilir
            race_data = result['race']
            if isinstance(race_data, dict):
                race_raw = max(race_data, key=race_data.get).lower()
            else:
                race_raw = race_data.lower()
            country = race_to_country.get(race_raw, race_raw.title())

            emotion = result['dominant_emotion']

            # Göstəricilər ekrana yazılır
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'{gender}, {age} yash', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f'{country}, {emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    except Exception as e:
        print("DeepFace analizi zamani xəta:", e)

    cv2.imshow('DeepFace Analiz', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
