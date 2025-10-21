from flask import Flask, render_template, request
from uuid import uuid4
from question_match import GraphQA
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求
graph_qa = GraphQA()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/welcome-1.html')
def index1():
    return render_template('welcome-1.html')

@app.route('/chat.html')
def index2():
    uuid = uuid4()
    return render_template('chat.html', uuid=uuid)
@app.route('/graph.html')
def index3():
    return render_template('graph.html')



@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    answer = graph_qa.query(data['question'], data['uuid'])
    return json.dumps({'answer': answer}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(debug=True)