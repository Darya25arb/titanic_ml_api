from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)


# загружаем сохраненную модель
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    # ожидаем, что данные придут в виде JSON
    data = request.get_json(force=True)
    # предполагаем, что JSON содержит необходимые признаки
    # например, {"Pclass": 3, "Sex": 0, "Age": 25, "Fare": 7.25, "Embarked": 1}
    try:
        features = np.array([
            data['Pclass'],
            data['Sex'],
            data['Age'],
            data['Fare'],
            data['Embarked']
        ]).reshape(1, -1)
    except KeyError as e:
        return jsonify({"error": f"Отсутствует ключ: {e}"}), 400
    
    # получаем предсказание
    prediction = model.predict(features)
    # возвращаем результат в виде JSON
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
