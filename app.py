from flask import Flask,request,jsonify
from flask_cors import CORS

from NER import NERClassifier

#global graph
#graph = tf.get_default_graph()

Ner = NERClassifier('./models/10_0.25_400_3_0.001_64_conllfff.h5', 30, 10)

app = Flask(__name__)
CORS(app)

@app.route("/predict",methods=['POST'])
def predict():
    
    text = request.json["text"]
    try:
        return Ner.predict(text)
    except Exception as e:
        print(e)
        return "Ha ocurrido un error"


if __name__ == "__main__":
    app.run('0.0.0.0',port=8000, threaded=False)