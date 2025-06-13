# Importar la aplicación Flask desde el archivo principal
from glucose_prediction_api import app

# Esta línea es necesaria para que Gunicorn encuentre la aplicación
if __name__ == "__main__":
    app.run(host='0.0.0.0')