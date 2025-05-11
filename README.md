# Modelo Predictivo de Glucosa

API de predicción de niveles de glucosa para pacientes con diabetes, basada en la ingesta de alimentos y datos del usuario.

## Características

- Predicción de niveles de glucosa post-prandial (después de comer)
- Recomendaciones personalizadas basadas en la predicción
- Estimación de confianza de la predicción
- Soporte para diferentes tipos de diabetes y comidas

## Tecnologías

- Flask para la API REST
- Scikit-learn para el modelo de machine learning
- Pandas y NumPy para procesamiento de datos
- Joblib para persistencia del modelo

## Endpoints

### GET /

Información general sobre la API.

### GET /health

Verificación del estado de la API.

### POST /predict

Realiza una predicción de glucosa basada en los datos proporcionados.

**Ejemplo de solicitud:**

```json
{
  "userId": "user123",
  "currentGlucose": 120,
  "foodEaten": "Arroz con pollo y ensalada",
  "foodMeal": "almuerzo",
  "userInfo": {
    "diabetesType": "Tipo 2",
    "age": 45,
    "weight": 70,
    "height": 170,
    "gender": "masculino"
  }
}
```

**Ejemplo de respuesta:**

```json
{
  "predictedGlucose": 145,
  "confidence": 0.85,
  "timeframe": "2 horas",
  "recommendations": [
    "Se espera un aumento moderado de glucosa. Mantén un equilibrio entre proteínas y carbohidratos.",
    "Incluir verduras y ensaladas en el almuerzo ayuda a ralentizar la absorción de carbohidratos.",
    "Combinar actividad física con una alimentación equilibrada ayuda a mejorar la sensibilidad a la insulina.",
    "Una caminata corta después de comer puede ayudar a reducir el pico de glucosa."
  ]
}
```

## Instalación y ejecución local

1. Clonar el repositorio:
```bash
git clone https://github.com/FabriCrespo/ModeloPredictivoGlucosa.git
cd ModeloPredictivoGlucosa
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar la aplicación:
```bash
python glucose_prediction_api.py
```

La API estará disponible en http://localhost:5000

## Despliegue

Este proyecto está configurado para despliegue en Render o Heroku.

### Render

Simplemente conecta este repositorio a Render y usa la configuración en `render.yaml`.

### Heroku

```bash
heroku create
git push heroku main
```

## Licencia

MIT