import threading
import json

from config.ServerConfig import *
from utils.BotServer import BotServer
from models.InsideOut import *

def to_client(conn, addr, model_dict):
    try:
        # 데이터 수신
        read = conn.recv(2048)  # 수신 데이터가 있을 때 까지 블로킹
        print('===========================')
        print('Connection from: %s' % str(addr))

        if read is None or not read:
            # 클라이언트 연결이 끊어지거나, 오류가 있는 경우
            print('클라이언트 연결 끊어짐')
            exit(0)

        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']
        bot_type = recv_json_data['BotType']

        # 모델 돌려서 answer 얻기
        model = model_dict[bot_type]
        try:
            answer = model.predict(query)
        except:
            answer = "에러 났어요 삐용삐용"

        answer_image = None # 원하면 넣기

        send_json_data_str = {
            "Query" : query,
            "Answer": answer,
            "AnswerImageUrl" : answer_image
        }
        message = json.dumps(send_json_data_str)
        conn.send(message.encode())

    except Exception as ex:
        print(ex)

    finally:
        conn.close()


if __name__ == '__main__':

    model_dict = dict()
    model_dict['ANGER'] = InsideOut('ANGER')
    model_dict['JOY'] = InsideOut('JOY')
    model_dict['SADNESS'] = InsideOut('SADNESS')
    model_dict['BINGBONG'] = InsideOut('BINGBONG')
    print("model completed")

    port = ENGINE_PORT
    listen = 100

    # 봇 서버 동작
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start")

    while True:
        conn, addr = bot.ready_for_client()
        client = threading.Thread(target=to_client, args=(
            conn,
            addr,
            model_dict
        ))
        client.start()