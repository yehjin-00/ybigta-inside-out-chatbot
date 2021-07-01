from config.ServerConfig import *
from utils.BotServer import BotServer
from models.InsideOut import *
from models.Emotion import *
from models.Modelling import *
import threading
import json
from silence_tensorflow import silence_tensorflow

# tensorflow warning 안 나오게 하기
silence_tensorflow()

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
        bot_type = recv_json_data['BotType']
        query = recv_json_data['Query']


        # 모델 돌려서 answer 얻기
        model = model_dict[bot_type]
        try:
            if bot_type == BOT_TYPE[0]: # EMOTION
                answer_image = model.predict(query)
                answer = None
            else:
                answer = model.predict(query)
                answer_image = None

        except:
            answer = "에러 났어요 삐용삐용"

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

    print("MODEL START")
    model_dict = dict()

    # 감정 모델
    print(f"1. {BOT_TYPE[0].lower()} model start")
    model_dict[BOT_TYPE[0]] = Emotion()

    for i, bot in enumerate(BOT_TYPE[1:]):
        print(f"{i+2}. {bot.lower()} model start")
        if bot == 'BINGBONG':
            model_dict[bot] = InsideOut(bot, 2)
        else:
            model_dict[bot] = InsideOut(bot, 6)

    print("MODEL COMPLETED")

    port = ENGINE_PORT
    listen = 100

    # 봇 서버 동작
    bot = BotServer(port, listen)
    bot.create_sock()
    print("BOT START")

    while True:
        conn, addr = bot.ready_for_client()
        client = threading.Thread(target=to_client, args=(
            conn,
            addr,
            model_dict
        ))
        client.start()