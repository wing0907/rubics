"""
모바일 접근 안내 페이지 제공
포트 8000에서 실행
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import threading
import time

class QubeHTTPHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/mobile_access.html'
        return SimpleHTTPRequestHandler.do_GET(self)
    
    def log_message(self, format, *args):
        print(f"[HTTP] {format % args}")

def start_server():
    os.chdir(os.path.dirname(__file__))
    server = HTTPServer(('0.0.0.0', 8000), QubeHTTPHandler)
    print(f"📱 모바일 안내 페이지: http://10.1.0.59:8000")
    print(f"🚀 Streamlit MVP: http://10.1.0.59:8501")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료")
        server.server_close()

if __name__ == '__main__':
    start_server()
