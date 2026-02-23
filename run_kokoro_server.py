import uvicorn
import socket
import argparse
import sys

def find_free_port(start_port=8000, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def main():
    parser = argparse.ArgumentParser(description='Run Kokoro API server')
    parser.add_argument('--port', type=int, default=8001, help='Port to run the server on')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to run the server on')
    parser.add_argument('--auto-port', action='store_true', help='Automatically find an available port')
    args = parser.parse_args()

    port = args.port
    if args.auto_port or port == 8001:
        available_port = find_free_port(start_port=port)
        if available_port is None:
            print(f"Error: Could not find an available port starting from {port}")
            sys.exit(1)
        if available_port != port:
            print(f"Port {port} is in use, using port {available_port} instead")
            port = available_port

    print(f"Starting server on http://{args.host}:{port}")
    print(f"Documentation available at:")
    print(f"  - http://{args.host}:{port}/docs")
    print(f"  - http://{args.host}:{port}/redoc")

    uvicorn.run(
        "tts_server:app",
        host=args.host,
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main()
