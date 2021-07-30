# bot types (무조건 EMOTION이 처음)
BOT_TYPE = ['EMOTION', 'ANGER', 'JOY', 'SADNESS', 'FEAR', 'BINGBONG']

# train data dict (use bot_type for key)
TRAIN_DATA = dict()

# checkpoint dict (use bot_type for key)
CHECKPOINT = dict()

for bot in BOT_TYPE:
    TRAIN_DATA[bot] = f'/home/ubuntu/ybigta-inside-out-chatbot/chatbot/models/dataset/{bot.lower()}_train_data.csv'
    CHECKPOINT[bot] = f'/home/ubuntu/ybigta-inside-out-chatbot/chatbot/models/checkpoint/{bot.lower()}/cp.ckpt'

CHECKPOINT['EMOTION'] = f'/home/ubuntu/ybigta-inside-out-chatbot/chatbot/models/checkpoint/emotion/model_final_model.pth'

def FilePathConfig():
    global TRAIN_DATA, CHECKPOINT, BOT_TYPE