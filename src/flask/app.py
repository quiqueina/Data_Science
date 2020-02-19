#!flask/bin/python
from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/home', methods = ['POST', 'GET'])
def index():

    if request.method == 'GET':

        return "Hello, kike!"
    elif request.method == 'POST':
        return 'Hola Cresko'
    else:
        return {"status_code": 404, 'message': "not allowed"}


if __name__ == '__main__':
    app.run(debug=True)
