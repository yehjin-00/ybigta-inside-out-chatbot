from flask import Flask, request, jsonify, abort
from config.ServerConfig import *
import socket
import json

# Flask 어플리케이션
app = Flask(__name__)

# 챗봇 엔진 서버와 통신
def get_answer_from_engine(bottype, query):
    # 챗봇 엔진 서버 연결
    mySocket = socket.socket()
    mySocket.connect((ENGINE_HOST, ENGINE_PORT))

    # 챗봇 엔진 질의 요청
    json_data = {
        'Query': query,
        'BotType': bottype
    }
    message = json.dumps(json_data)
    mySocket.send(message.encode())

    # 챗봇 엔진 답변 출력
    data = mySocket.recv(2048).decode()
    ret_data = json.loads(data)

    # 챗봇 엔진 서버 연결 소켓 닫기
    mySocket.close()

    return ret_data


@app.route('/', methods=['GET'])
def index():
    print('hello')


# 챗봇 엔진 query 전송 API
@app.route('/query/<bot_type>', methods=['POST'])
def query(bot_type):
    body = request.get_json()

    try:
        BOT_TYPE = ['BINGBONG', 'ANGER', 'JOY', 'SADNESS']

        # BOT_TYPE에 해당되지 않으면 404 오류
        if bot_type not in BOT_TYPE:
            abort(404)

        # BOT_TYPE에 해당되면 작동!
        else:
            # 카카오톡 처리
            body = request.get_json()
            utterance = body['userRequest']['utterance']
            ret = get_answer_from_engine(bottype=bot_type, query=utterance)

            from chatbot_api.KakaoTemplate import KakaoTemplate
            skillTemplate = KakaoTemplate()
            return skillTemplate.send_response(ret)

    except Exception as ex:
        # 오류 발생시 500 오류
        abort(500)

if __name__ == '__main__':
    # API_HOST, API_PORT는 config > ServerConfig 에서 정의
    app.run(host=API_HOST, port=API_PORT)
