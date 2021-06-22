# models folder directory
PATH = '/home/ubuntu/tmp/pycharm_project_880/models/'

# train data dict (use bot_type for key)
TRAIN_DATA = dict()
TRAIN_DATA['ANGER'] = PATH + 'dataset/anger_train_data.csv'
TRAIN_DATA['SADNESS'] = PATH + 'dataset/sadness_train_data.csv'
TRAIN_DATA['JOY'] = PATH + 'dataset/joy_train_data.csv'
TRAIN_DATA['BINGBONG'] = PATH + 'dataset/bingbong_train_data.csv'

# checkpoint dict (use bot_type for key)
CHECKPOINT = dict()
CHECKPOINT['ANGER'] = PATH + 'checkpoint/anger/cp.ckpt'
CHECKPOINT['SADNESS'] = PATH + 'checkpoint/sadness/cp.ckpt'
CHECKPOINT['JOY'] = PATH + 'checkpoint/joy/cp.ckpt'
CHECKPOINT['BINGBONG'] = PATH + 'checkpoint/bingbong/cp.ckpt'

def ServerConfig():
    global TRAIN_DATA, CHECKPOINT