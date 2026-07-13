import os
import sys
import webbrowser
import threading
import json
import base64
import subprocess
import sounddevice as sd
import soundfile as sf
import pyttsx3

# Thread lock to prevent concurrent SAPI5 initializations
tts_lock = threading.Lock()

PORT = 8000

def get_cable_device_index():
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            if dev['max_output_channels'] > 0 and ('cable input' in name or 'virtual audio' in name or 'vb-audio' in name):
                return i
    except Exception as e:
        print("Error querying audio devices:", e)
    return None

def run_flask():
    try:
        from flask import Flask, send_from_directory, request, jsonify
        
        app = Flask(__name__, static_folder='.')
        
        @app.route('/')
        def serve_index():
            return send_from_directory('.', 'index.html')
            
        @app.route('/<path:path>')
        def serve_static(path):
            return send_from_directory('.', path)
            
        @app.route('/api/status')
        def status():
            return jsonify({"status": "online", "message": "SignCV Python API active"})
            
        @app.route('/api/devices', methods=['GET'])
        def get_devices():
            cable_idx = get_cable_device_index()
            if cable_idx is not None:
                devices = sd.query_devices()
                return jsonify({
                    "connected": True,
                    "device_index": cable_idx,
                    "device_name": devices[cable_idx]['name']
                })
            return jsonify({"connected": False})
            
        @app.route('/api/speak', methods=['POST'])
        def speak():
            data = request.get_json() or {}
            text = data.get('text', '').strip()
            if not text:
                return jsonify({"success": False, "error": "No text provided"}), 400
                
            try:
                cable_idx = get_cable_device_index()
                if cable_idx is None:
                    return jsonify({"success": False, "error": "Virtual Microphone (CABLE Input) not found. Please install VB-CABLE."}), 404
                    
                with tts_lock:
                    temp_wav = "temp_speech.wav"
                    engine = pyttsx3.init()
                    engine.save_to_file(text, temp_wav)
                    engine.runAndWait()
                    
                    audio_data, fs = sf.read(temp_wav)
                    sd.play(audio_data, fs, device=cable_idx)
                    sd.wait()
                    
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
                        
                return jsonify({"success": True})
            except Exception as e:
                print("TTS Playback Error:", e)
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/api/deploy-model', methods=['POST'])
        def deploy_model():
            data = request.get_json() or {}
            model_topology = data.get('modelTopology')
            weight_specs = data.get('weightSpecs')
            weights_base64 = data.get('weightsBase64')
            labels = data.get('labels')
            
            if not (model_topology and weight_specs and weights_base64 and labels):
                return jsonify({"success": False, "error": "Invalid model payload structure"}), 400
                
            try:
                # 1. Create model output directory
                os.makedirs('model', exist_ok=True)
                
                # 2. Write model.json
                model_json = {
                    "modelTopology": model_topology,
                    "weightsManifest": [{
                        "paths": ["model.weights.bin"],
                        "weights": weight_specs
                    }]
                }
                with open('model/model.json', 'w') as f:
                    json.dump(model_json, f, indent=2)
                    
                # 3. Write labels.json
                with open('model/labels.json', 'w') as f:
                    json.dump(labels, f, indent=2)
                    
                # 4. Write binary weights
                weights_bin = base64.b64decode(weights_base64)
                with open('model/model.weights.bin', 'wb') as f:
                    f.write(weights_bin)
                    
                # 5. Git commit & Push to GitHub remote
                def run_git(args):
                    res = subprocess.run(args, capture_output=True, text=True, check=True)
                    return res.stdout.strip()
                
                git_msg = "Model compiled and saved locally."
                try:
                    run_git(['git', 'add', 'model/model.json', 'model/model.weights.bin', 'model/labels.json'])
                    run_git(['git', 'commit', '-m', 'Auto-deploy trained model from Developer Trainer'])
                    git_push_out = run_git(['git', 'push', 'origin', 'main'])
                    git_msg = git_push_out or "Model committed and pushed to remote main branch."
                except Exception as g_err:
                    git_msg = f"Model saved locally, but Git push warning: {str(g_err)}"
                    print(git_msg)
                
                return jsonify({
                    "success": True,
                    "git_output": git_msg
                })
            except Exception as e:
                print("Deploy model error:", e)
                return jsonify({"success": False, "error": str(e)}), 500

        @app.route('/api/push-dataset', methods=['POST'])
        def push_dataset():
            data = request.get_json() or {}
            dataset = data.get('dataset')
            if not dataset:
                return jsonify({"success": False, "error": "No dataset payload found"}), 400
                
            try:
                # Save to root folder
                with open('asl_custom_dataset.json', 'w') as f:
                    json.dump(dataset, f, indent=2)
                    
                # Commit and push
                def run_git(args):
                    res = subprocess.run(args, capture_output=True, text=True, check=True)
                    return res.stdout.strip()
                    
                git_msg = "Dataset compiled and saved locally."
                try:
                    run_git(['git', 'add', 'asl_custom_dataset.json'])
                    run_git(['git', 'commit', '-m', 'Auto-deploy raw gesture dataset from Developer Portal'])
                    git_push_out = run_git(['git', 'push', 'origin', 'main'])
                    git_msg = git_push_out or "Dataset pushed successfully to remote main branch."
                except Exception as g_err:
                    git_msg = f"Dataset saved locally, but Git push warning: {str(g_err)}"
                    print(git_msg)
                
                return jsonify({
                    "success": True,
                    "git_output": git_msg
                })
            except Exception as e:
                print("Push dataset error:", e)
                return jsonify({"success": False, "error": str(e)}), 500
            
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
            
        def do_GET(self):
            if self.path == '/api/devices':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                cable_idx = get_cable_device_index()
                if cable_idx is not None:
                    devices = sd.query_devices()
                    res = {
                        "connected": True,
                        "device_index": cable_idx,
                        "device_name": devices[cable_idx]['name']
                    }
                else:
                    res = {"connected": False}
                self.wfile.write(json.dumps(res).encode('utf-8'))
                return
            elif self.path == '/api/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "online", "message": "SignCV Python API active"}).encode('utf-8'))
                return
                
            super().do_GET()
            
        def do_POST(self):
            if self.path == '/api/speak':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                except:
                    data = {}
                text = data.get('text', '').strip()
                
                if not text:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "No text provided"}).encode('utf-8'))
                    return
                    
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                try:
                    cable_idx = get_cable_device_index()
                    if cable_idx is None:
                        res = {"success": False, "error": "Virtual Microphone (CABLE Input) not found. Please install VB-CABLE."}
                    else:
                        with tts_lock:
                            temp_wav = "temp_speech.wav"
                            engine = pyttsx3.init()
                            engine.save_to_file(text, temp_wav)
                            engine.runAndWait()
                            
                            audio_data, fs = sf.read(temp_wav)
                            sd.play(audio_data, fs, device=cable_idx)
                            sd.wait()
                            
                            if os.path.exists(temp_wav):
                                os.remove(temp_wav)
                        res = {"success": True}
                except Exception as e:
                    print("TTS Playback Error:", e)
                    res = {"success": False, "error": str(e)}
                    
                self.wfile.write(json.dumps(res).encode('utf-8'))
                return
                
            elif self.path == '/api/deploy-model':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                except:
                    data = {}
                    
                model_topology = data.get('modelTopology')
                weight_specs = data.get('weightSpecs')
                weights_base64 = data.get('weightsBase64')
                labels = data.get('labels')
                
                if not (model_topology and weight_specs and weights_base64 and labels):
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "Invalid model payload structure"}).encode('utf-8'))
                    return
                    
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                try:
                    os.makedirs('model', exist_ok=True)
                    model_json = {
                        "modelTopology": model_topology,
                        "weightsManifest": [{
                            "paths": ["model.weights.bin"],
                            "weights": weight_specs
                        }]
                    }
                    with open('model/model.json', 'w') as f:
                        json.dump(model_json, f, indent=2)
                        
                    with open('model/labels.json', 'w') as f:
                        json.dump(labels, f, indent=2)
                        
                    weights_bin = base64.b64decode(weights_base64)
                    with open('model/model.weights.bin', 'wb') as f:
                        f.write(weights_bin)
                        
                    def run_git(args):
                        res = subprocess.run(args, capture_output=True, text=True, check=True)
                        return res.stdout.strip()
                        
                    git_msg = "Model compiled and saved locally."
                    try:
                        run_git(['git', 'add', 'model/model.json', 'model/model.weights.bin', 'model/labels.json'])
                        run_git(['git', 'commit', '-m', 'Auto-deploy trained model from Developer Trainer'])
                        git_push_out = run_git(['git', 'push', 'origin', 'main'])
                        git_msg = git_push_out or "Model committed and pushed to remote main branch."
                    except Exception as g_err:
                        git_msg = f"Model saved locally, but Git push warning: {str(g_err)}"
                        print(git_msg)
                        
                    res = {
                        "success": True,
                        "git_output": git_msg
                    }
                except Exception as e:
                    res = {"success": False, "error": str(e)}
                    
                self.wfile.write(json.dumps(res).encode('utf-8'))
                return
                
            elif self.path == '/api/push-dataset':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                except:
                    data = {}
                dataset = data.get('dataset')
                
                if not dataset:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "No dataset payload found"}).encode('utf-8'))
                    return
                    
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                try:
                    with open('asl_custom_dataset.json', 'w') as f:
                        json.dump(dataset, f, indent=2)
                        
                    def run_git(args):
                        res = subprocess.run(args, capture_output=True, text=True, check=True)
                        return res.stdout.strip()
                        
                    git_msg = "Dataset compiled and saved locally."
                    try:
                        run_git(['git', 'add', 'asl_custom_dataset.json'])
                        run_git(['git', 'commit', '-m', 'Auto-deploy raw gesture dataset from Developer Portal'])
                        git_push_out = run_git(['git', 'push', 'origin', 'main'])
                        git_msg = git_push_out or "Dataset pushed successfully to remote main branch."
                    except Exception as g_err:
                        git_msg = f"Dataset saved locally, but Git push warning: {str(g_err)}"
                        print(git_msg)
                        
                    res = {
                        "success": True,
                        "git_output": git_msg
                    }
                except Exception as e:
                    res = {"success": False, "error": str(e)}
                    
                self.wfile.write(json.dumps(res).encode('utf-8'))
                return
                
            super().do_POST()

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
