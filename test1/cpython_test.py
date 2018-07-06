from socketserver import ForkingTCPServer
from socketserver import StreamRequestHandler

server = ForkingTCPServer("127.0.0.1:8888", StreamRequestHandler)
server.serve_forever()