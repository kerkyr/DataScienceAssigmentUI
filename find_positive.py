import itertools, random, os, zipfile
import pandas as pd
from joblib import load

# ---------------- Load model ----------------
zip_path = 'best_rf_model.zip'
unzip_dir = 'model/'

# Unzip if needed
if not os.path.exists(unzip_dir):
    os.makedirs(unzip_dir)

model_path = os.path.join(unzip_dir, 'best_rf_model.joblib')
if not os.path.exists(model_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)

model = load(model_path)

# ---------------- Config ----------------
THRESHOLD = 0.25   # change to 0.30 if you want to be more sensitive
classes = list(getattr(model, "classes_", []))
print("Model classes_:", classes)

# pick positive-class column for predict_proba
if 1 in classes:
    pos_idx = classes.index(1)
    pos_label = 1
elif "Yes" in classes:
    pos_idx = classes.index("Yes")
    pos_label = "Yes"
else:
    pos_idx = len(classes) - 1
    pos_label = classes[pos_idx]
print(f"[INFO] Using positive class: {pos_label} (proba column {pos_idx})")
print(f"[INFO] Decision threshold: {THRESHOLD:.2f}")

# ---------------- Helpers ----------------
def make_row(age, bmi, chol, sleep, stress, sugar, exer, male, smoke_yes, hbp_yes, fbs_high):
    return {
        "Fasting Blood Sugar": 1 if fbs_high else 0,
        "BMI": float(bmi),
        "Cholesterol Level": float(chol),
        "Sleep Hours": float(sleep),
        "Age": int(age),
        "Stress Level": int(stress),            # 0/1/2
        "Sugar Consumption": int(sugar),        # 0/1/2
        "Exercise Habits": int(exer),           # 0/1/2
        "Gender_Male": int(male),               # 0/1
        "Smoking_Yes": int(smoke_yes),          # 0/1
        "High Blood Pressure_Yes": int(hbp_yes) # 0/1
    }

best = {"proba": -1.0, "row": None, "source": ""}

try:
    # ---------------- Coarse grid search ----------------
    print("[PHASE] Grid search...")
    ages = [30, 45, 60, 75]
    bmis = [18, 25, 32, 38]
    chols = [180, 220, 260, 300, 340]
    sleeps = [3, 5, 7]
    ords = [0, 1, 2]
    bin01 = [0, 1]

    for age, bmi, chol, sleep, stress, sugar, exer, male, smoke, hbp, fbs in itertools.product(
        ages, bmis, chols, sleeps, ords, ords, ords, bin01, bin01, bin01, bin01
    ):
        row = make_row(age, bmi, chol, sleep, stress, sugar, exer, male, smoke, hbp, fbs)
        proba = float(model.predict_proba(pd.DataFrame([row]))[0, pos_idx])
        if proba > best["proba"]:
            best = {"proba": proba, "row": row, "source": "grid"}
        if proba >= THRESHOLD:
            print("\n✅ Found positive example (>= threshold) in GRID:")
            print(f"Probability = {proba:.3f}")
            print(pd.Series(row))
            break
    else:
        # ---------------- Random search (wider sweep) ----------------
        print("[PHASE] Random search (3000 trials)...")
        for t in range(1, 3001):
            row = make_row(
                age=random.randint(25, 85),
                bmi=round(random.uniform(16, 45), 1),
                chol=round(random.uniform(150, 380), 0),
                sleep=round(random.uniform(2, 9), 1),
                stress=random.randint(0, 2),
                sugar=random.randint(0, 2),
                exer=random.randint(0, 2),
                male=random.randint(0, 1),
                smoke_yes=random.randint(0, 1),
                hbp_yes=random.randint(0, 1),
                fbs_high=random.randint(0, 1),
            )
            proba = float(model.predict_proba(pd.DataFrame([row]))[0, pos_idx])
            if proba > best["proba"]:
                best = {"proba": proba, "row": row, "source": "random"}
            if proba >= THRESHOLD:
                print("\n✅ Found positive example (>= threshold) in RANDOM:")
                print(f"Probability = {proba:.3f}")
                print(pd.Series(row))
                break
            if t % 500 == 0:
                print(f"[INFO] Random tried: {t}, current best proba = {best['proba']:.3f}")

    # ---------------- Report best found (always) ----------------
    if best["row"] is not None:
        print("\n[RESULT] Highest probability found = "
              f"{best['proba']:.3f}  (source: {best['source']})")
        print("Input features for this case:")
        print(pd.Series(best["row"]))
    else:
        print("\n[RESULT] No candidates evaluated (check model/feature names).")

finally:
    print("\ncomplete")
