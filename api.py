from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
import os
import numpy as np
from sklearn.externals import joblib


path = os.path.dirname(os.path.realpath(__file__))
analytics_path = os.path.join(path, 'analytics')
iris_model_file = os.path.join(analytics_path, 'iris_model.pkl')

# si estas sentencias se ponen acá, los objetos se cargan en la memoria del servidor al momento de iniciar la API,
# esto hace que las respuestas a las peticiones sean más rápidas
# si se ponen dentro de la función o clase, los objetos se leen del disco cada vez que se invoca el endpoint,
# lo que es más lento, pero reduce el uso de memoria del servidor
iris_model = joblib.load(iris_model_file)


app = Flask(__name__)
api = Api(app)


# ## UI para pruebas

@app.route('/', methods=['GET'])
def tester():
    """Esta vista genera un formulario para probar la API"""
    return render_template('tester.html')


# ## pronósticos

class Iris(Resource):

    def scorer(self, data, classifier):
        try:
            sepal_length, sepal_width = float(data['sepal_length']), float(data['sepal_width'])
            petal_length, petal_width = float(data['petal_length']), float(data['petal_width'])
            # deben ir en el mismo orden en que se estimó el modelo
            data_p = [[sepal_length, sepal_width, petal_length, petal_width]]
            pred = classifier.predict(data_p)[0]
            ppred = round(np.max(classifier.predict_proba(data_p)), 4)
            return jsonify(predicted=pred, probability=float(ppred), message='success')
        except:
            return jsonify(predicted=None, probability=None, message='There was an error')

    def get(self):
        reqdata = request.args
        result = self.scorer(data=reqdata, classifier=iris_model)
        return result

    def post(self):
        reqdata = request.get_json()
        result = self.scorer(data=reqdata, classifier=iris_model)
        return result


# ## endpoints

api.add_resource(Iris, '/iris')


if __name__ == '__main__':
    app.run(debug=True)
