# bot types (무조건 EMOTION이 마지막)
BOT_TYPE = ['EMOTION', 'ANGER', 'JOY', 'SADNESS']

# train data dict (use bot_type for key)
TRAIN_DATA = dict()

# checkpoint dict (use bot_type for key)
CHECKPOINT = dict()

for bot in BOT_TYPE:
    TRAIN_DATA[bot] = f'/home/ubuntu/pycharm/models/dataset/{bot.lower()}_train_data.csv'
    CHECKPOINT[bot] = f'/home/ubuntu/pycharm/models/checkpoint/{bot.lower()}/cp.ckpt'

CHECKPOINT['EMOTION'] = f'/home/ubuntu/pycharm/models/checkpoint/emotion/model_final_model.pth'

def ServerConfig():
    global TRAIN_DATA, CHECKPOINT, BOT_TYPE