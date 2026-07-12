import os
import sys
import webbrowser

# Dual-mode server: Try to run Flask, fallback to standard library http.server if Flask is not installed
PORT = 8000

def run_flask():
    try:
        from flask import Flask, send_from_directory
        
        app = Flask(__name__, static_folder='.')
        
        @app.route('/')
        def serve_index():
            return send_from_directory('.', 'index.html')
            
        @app.route('/<path:path>')
        def serve_static(path):
            return send_from_directory('.', path)
            
        @app.route('/api/status')
        def status():
            return {"status": "online", "message": "SignCV Python API active"}
            
        print("*" * 50)
        print(f"Starting Flask Server on http://localhost:{PORT}")
        print("*" * 50)
        
        # Open web browser automatically
        webbrowser.open(f"http://localhost:{PORT}")
        app.run(host='0.0.0.0', port=PORT, debug=False)
        return True
    except ImportError:
        return False

def run_http_server():
    import http.server
    import socketserver
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Support CORS and disable caching during dev
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            super().end_headers()

    print("*" * 50)
    print("Flask not found. Falling back to built-in Python HTTP Server...")
    print(f"Serving static files on http://localhost:{PORT}")
    print("*" * 50)
    
    # Open web browser automatically
    webbrowser.open(f"http://localhost:{PORT}")
    
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            sys.exit(0)

if __name__ == '__main__':
    # Print welcome logo
    print("""
    ==================================================
      SignCV - Sign Language Recognition Server
    ==================================================
    """)
    if not run_flask():
        run_http_server()
