from socketserver import ForkingTCPServer
from socketserver import StreamRequestHandler


class Handler(StreamRequestHandler):

    def handle(self):
       	request = self.rfile.read(1024)
        print(request.decode())
        http_response = b"""\
HTTP/1.1 200 OK

Hello, World!
"""
        self.wfile.write(http_response)


server = ForkingTCPServer(("127.0.0.1", 8888), Handler)
server.serve_forever()
