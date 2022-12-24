from flask import Flask
from flask import request
from flask_cors import CORS

app = Flask(__name__)


@app.route('/')
def hello_world():
    page = request.args.get('page', default=1, type=int)
    return 'Hello, World!' + str(page)


CORS(app.run(ssl_context=('fullchain.pem', 'privkey.pem'), host='0.0.0.0', port=5000))
