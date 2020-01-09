import os
import logging
import coloredlogs
import socket

DEBUG =0
#SERVER_IP = '127.0.0.1'
SERVER_IP = '192.168.0.20'
#SERVER_IP = '158.182.9.51' # 1 node 8 workers
#SERVER_IP = '158.182.9.51' # 2 nodes 16 workers
#SERVER_IP = '158.182.9.53'  # 4 nodes 32 workers
#SERVER_IP = '158.182.9.50'  # 8 nodes 64 workers
#SERVER_IP = '158.182.9.78'  # 8 nodes 64 workers
#SERVER_IP = '158.182.9.40' # 16 node 128 workers
#SERVER_IP = '158.182.9.50'  # 32 nodes 256 workers
SERVER_PORT = 5911
PORT = 5922
ACTIVE_WAIT=False
PASSIVE_WAIT=False
GPU_CONSTRUCTION=True
SPARSE=False
WARMUP=True
ZHU=False
PS=False
if PS:
    ACTIVE_WAIT=True
DELAY_COMM=1
if SPARSE:
    PREFIX='compression'
else:
    PREFIX='baseline'
if WARMUP:
    PREFIX=PREFIX+'-gwarmup'
if ACTIVE_WAIT:
    PREFIX=PREFIX+'-wait'
if PS:
    PREFIX=PREFIX+'-ps'
#if DELAY_COMM > 1:
PREFIX=PREFIX+'-dc'+str(DELAY_COMM)
EXCHANGE_MODE = 'TOPK_MODEL' 
#EXCHANGE_MODE = 'MODEL' 
#EXCHANGE_MODE = 'MODEL+GRAD' 
#EXCHANGE_MODE = 'GRAD' 
#PREFIX=PREFIX+'-'+EXCHANGE_MODE.lower()+'-ijcai-fixedlr'
PREFIX=PREFIX+'-'+EXCHANGE_MODE.lower()+'-ijcai-debug2'
if ZHU:
    PREFIX=PREFIX+'-zhu'
DELAY=0

MAX_EPOCHS = 200
#MAX_EPOCHS = 90

BOOTSTRAP_LIST = ['gpu10', 'gpu11']#, 'gpu12', 'gpu13']
#BOOTSTRAP_LIST = ['gpu10', 'gpu11', 'gpu12', 'gpu13', 'gpu14', 'gpu15', 'gpu16', 'gpu17', 'gpu18', 'gpu19']

hostname = socket.gethostname() 
logger = logging.getLogger(hostname)

if DEBUG:
    #coloredlogs.install(level='DEBUG')
    logger.setLevel(logging.DEBUG)
else:
    #coloredlogs.install(level='INFO')
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

