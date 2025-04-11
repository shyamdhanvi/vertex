from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

with open("heart_disease_model_compact_inputs.sav", "rb") as f:
    model = pickle.load(f)

snps = ['rs123', 'rs456', 'rs789', 'rs101', 'rs102', 'rs103', 'rs104', 'rs105', 'rs106', 'rs107']
genotypes = ['AA', 'AG', 'GG', 'TT', 'CC', 'TC', 'AT']
required_features = list(model.feature_names_in_)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = {
            "BMI": float(data.get("BMI", 0)),
            "Smoking": int(data.get("Smoking", 0)),
            "PhysicalHealth": int(data.get("PhysicalHealth", 0)),
            "Diabetic": int(data.get("Diabetic", 0))
        }

        age_categories = [
            "AgeCategory_55-59", "AgeCategory_60-64", "AgeCategory_65-69",
            "AgeCategory_70-74", "AgeCategory_75-79", "AgeCategory_80 or older"
        ]
        for cat in age_categories:
            input_data[cat] = int(data.get(cat, 0))

        gen_health = ["GenHealth_Fair", "GenHealth_Good", "GenHealth_Poor"]
        for g in gen_health:
            input_data[g] = int(data.get(g, 0))

        prs = 0
        snp_features = {}
        for snp in snps:
            user_geno = data.get(snp, "")
            for g in genotypes:
                snp_key = f"{snp}_{g}"
                snp_features[snp_key] = 1 if user_geno == g else 0
            prs += 2.5 if user_geno else 0

        input_data["PRS"] = prs
        input_data["GeneticRisk"] = prs / len(snps)
        input_data.update(snp_features)

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=required_features, fill_value=0)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        result = "Positive" if prediction == 1 else "Negative"

        return jsonify({
            "result": result,
            "probability": round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
