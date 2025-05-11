import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime, timedelta
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Ruta para guardar/cargar el modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'glucose_model.joblib')

# Características para el modelo
FEATURES = [
    'current_glucose', 'carbohydrates', 'glycemic_index', 
    'diabetes_type', 'time_of_day', 'hours_since_last_meal'
]

# Cargar o crear el modelo
def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        print("Cargando modelo existente...")
        return joblib.load(MODEL_PATH)
    else:
        print("Creando nuevo modelo...")
        # Crear un modelo básico inicial
        # Preprocesamiento para características numéricas y categóricas
        numeric_features = ['current_glucose', 'carbohydrates', 'glycemic_index', 'hours_since_last_meal']
        categorical_features = ['diabetes_type', 'time_of_day']
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Crear pipeline con preprocesamiento y modelo
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Datos de ejemplo para entrenar el modelo inicial
        X_sample = pd.DataFrame({
            'current_glucose': [120, 150, 180, 100, 130],
            'carbohydrates': [50, 30, 60, 20, 40],
            'glycemic_index': [70, 50, 80, 40, 60],
            'diabetes_type': ['Tipo 1', 'Tipo 2', 'Tipo 1', 'Tipo 2', 'Tipo 1'],
            'time_of_day': ['mañana', 'tarde', 'noche', 'mañana', 'tarde'],
            'hours_since_last_meal': [2, 4, 1, 3, 5]
        })
        
        # Valores objetivo de ejemplo
        y_sample = np.array([140, 160, 200, 110, 150])
        
        # Entrenar el modelo con datos de ejemplo
        model.fit(X_sample, y_sample)
        
        # Guardar el modelo
        try:
            joblib.dump(model, MODEL_PATH)
        except Exception as e:
            print(f"No se pudo guardar el modelo: {e}")
        
        return model

# Inicializar modelo
model = None

# Preparar datos de entrada para el modelo
def prepare_input_data(data):
    # Extraer valores o usar valores predeterminados
    current_glucose = data.get('currentGlucose', 120)
    food_meal = data.get('foodMeal', '').lower()
    food_eaten = data.get('foodEaten', '')
    diabetes_type = data.get('userInfo', {}).get('diabetesType', 'Tipo 2')
    
    # Estimar carbohidratos y índice glucémico basado en la comida
    carbs, gi = estimate_food_values(food_eaten, food_meal)
    
    # Determinar hora del día
    now = datetime.now()
    if now.hour < 12:
        time_of_day = 'mañana'
    elif now.hour < 18:
        time_of_day = 'tarde'
    else:
        time_of_day = 'noche'
    
    # Estimar horas desde última comida
    hours_since_meal = estimate_hours_since_meal(food_meal)
    
    # Crear DataFrame con los datos de entrada
    input_df = pd.DataFrame({
        'current_glucose': [current_glucose],
        'carbohydrates': [carbs],
        'glycemic_index': [gi],
        'diabetes_type': [diabetes_type],
        'time_of_day': [time_of_day],
        'hours_since_last_meal': [hours_since_meal]
    })
    
    return input_df

# Estimar valores nutricionales basados en la comida
def estimate_food_values(food_eaten, food_meal):
    # Valores predeterminados
    carbs = 30
    gi = 55
    
    # Ajustar según el tipo de comida
    if food_meal == 'desayuno':
        carbs = 40
        gi = 65
    elif food_meal == 'almuerzo':
        carbs = 60
        gi = 60
    elif food_meal == 'cena':
        carbs = 50
        gi = 55
    elif food_meal == 'merienda':
        carbs = 20
        gi = 50
    
    # Aquí se podría implementar un análisis más detallado del texto
    # de la comida para estimar mejor los valores nutricionales
    
    return carbs, gi

# Estimar horas desde última comida
def estimate_hours_since_meal(food_meal):
    # Valores predeterminados basados en el tipo de comida
    if food_meal == 'desayuno':
        return 10  # Asumiendo que la última comida fue la cena del día anterior
    elif food_meal == 'almuerzo':
        return 4   # Asumiendo ~4 horas desde el desayuno
    elif food_meal == 'cena':
        return 6   # Asumiendo ~6 horas desde el almuerzo
    elif food_meal == 'merienda':
        return 2   # Asumiendo ~2 horas desde la comida principal
    else:
        return 3   # Valor predeterminado

# Añadir rutas de la API
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "API de Predicción de Glucosa",
        "endpoints": {
            "/predict": "POST - Realizar predicción de glucosa",
            "/health": "GET - Verificar estado de la API"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    # Cargar modelo si no está cargado
    if model is None:
        model = load_or_create_model()
    
    # Obtener datos de la solicitud
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se proporcionaron datos"}), 400
    
    try:
        # Preparar datos para el modelo
        input_data = prepare_input_data(data)
        
        # Realizar predicción
        prediction = model.predict(input_data)[0]
        
        # Calcular confianza (simplificado)
        confidence = 0.85  # Valor de ejemplo
        
        # Generar recomendación basada en la predicción
        recommendation = generate_recommendation(prediction, data.get('currentGlucose', 120))
        
        return jsonify({
            "prediction": round(float(prediction), 1),
            "confidence": confidence,
            "recommendation": recommendation,
            "input_processed": {
                "glucose": input_data['current_glucose'][0],
                "carbs": input_data['carbohydrates'][0],
                "gi": input_data['glycemic_index'][0]
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_recommendation(predicted_glucose, current_glucose):
    if predicted_glucose > 180:
        return "La glucosa predicha es alta. Considera reducir la porción o elegir alimentos con menor índice glucémico."
    elif predicted_glucose < 70:
        return "La glucosa predicha es baja. Considera añadir más carbohidratos a tu comida."
    else:
        return "La glucosa predicha está en un rango adecuado."

# Para despliegue en Render
if __name__ == '__main__':
    # Cargar el modelo al iniciar
    model = load_or_create_model()
    
    # Obtener puerto del entorno (Render lo proporciona) o usar 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    
    # Ejecutar la aplicación
    app.run(host='0.0.0.0', port=port)
