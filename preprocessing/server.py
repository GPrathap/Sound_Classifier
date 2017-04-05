import Queue
import json
import socket
import sys
import threading

class UDPServer(threading.Thread):
    def __init__(self, threadID, input_buffer, port, receiver_port, ip="localhost"):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.isRun = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ip = ip
        self.port = port
        self.server_address = (self.ip,self.port)
        self.lock = threading.Lock()
        self.previous_buffer = [34, 67, 89]
        self.buffer = input_buffer
        self.receiver_port = receiver_port
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(self.server_address)
            print >>sys.stderr, 'starting up on %s port %s' % self.server_address
        except:
            print "Error: while starting up udp server"

    def run(self):
        print ("Starting " + self.threadID)
        self.call_back_handler()
        self.socket.close()
        print ("Exiting " + self.threadID)

    def get_previous_buffer(self):
        return self.previous_buffer

    def get_next_point(self):
        return self.previous_buffer
        #self.lock.acquire()
        # if not self.buffer.empty():
        #     data = self.buffer.get()
        #     #self.lock.release()
        #     return data
        #     #print "%s processing %s" % (threadName, data)
        # else:
        #     #self.lock.release()
        #     return self.previous_buffer

    def call_back_handler(self):
        while self.isRun:
            data, address = self.socket.recvfrom(self.receiver_port)
            data = json.loads(data)
            # print >>sys.stderr, 'received %s bytes from %s' % (len(data), address)
            # print >>sys.stderr, data
            # if data:
            self.lock.acquire()

            # self.buffer.put(data)
            self.previous_buffer = data
            # for i in self.buffer.get():
            #     print i
            self.lock.release()
            #     # sent = self.socket.sendto(data, address)
            #     # print >>sys.stderr, 'sent %s bytes back to %s' % (sent, address)




# queue = Queue.Queue(5)
# ip = "0.0.0.0"
# port = 8889
# receiver_port = 4096
# thread = UDPServer("udp_server", queue, port, receiver_port, ip)
# thread.start()
# thread.isRun = True
# thread.join()

