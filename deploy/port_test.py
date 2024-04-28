from flask import Flask, request, jsonify

app = Flask(__name__)

# 定义一个GET请求的接口
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, welcome to the server!'})

# 定义一个POST请求的接口
@app.route('/post_data', methods=['POST'])
def post_data():
    data = request.get_json()
    return jsonify({'received_data': data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
