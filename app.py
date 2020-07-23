from flask import Flask,request,jsonify
from flask_cors import CORS

from NER import NERClassifier

#global graph
#graph = tf.get_default_graph()

ancoraNer = NERClassifier('./models/crf_model_char_ancora_10.h5', 50, 50, corpus='ancora')
conllNer = NERClassifier('./models/10_0.25_400_3_0.001_64_conllfff.h5', 30, 10, corpus='conll')

app = Flask(__name__)
CORS(app)

@app.route("/predict",methods=['POST'])
def predict():
    
    text = request.json["text"]
    corpus = request.json['corpus']
    try:
        if corpus == 'conll':
            Ner = conllNer
        elif corpus == 'ancora':
            Ner = ancoraNer
        return Ner.predict(text)
    except Exception as e:
        print(e)
        return "Ha ocurrido un error"


if __name__ == "__main__":
    app.run('0.0.0.0',port=8000, threaded=False)