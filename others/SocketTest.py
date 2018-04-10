from socket import *
from time import ctime
HOST = ''
PORT =  12345
BUFSIZE = 1024
ADDR = (HOST,PORT)

tcpSocket = socket(AF_INET,SOCK_STREAM)
tcpSocket.bind(ADDR)
tcpSocket.listen(5)

while True:
    print('waiting for connectioin...')
    tcpClientSock ,addr = tcpSocket.accept()
    print('...connected from:',addr)
    while True:
        data = tcpClientSock.recv(BUFSIZE)
        if not data:
            break
        tcpClientSock.send(bytes(ctime(),encoding='utf-8'))
    tcpClientSock.close()
tcpSocket.close()