from flask import Flask, jsonify, request
from quote_training import return_quote 

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def get_quote():
    text = request.args.get('text')
    quote = return_quote(str(text))
    return jsonify({'quote': quote})


if __name__ == '__main__':  
    app.run(debug = True)