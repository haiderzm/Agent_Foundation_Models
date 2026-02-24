#!/usr/bin/env bash

# =====================================
# PaddleOCR FastAPI Server Controller
# =====================================

APP="paddle_ocr_server:app"
HOST="127.0.0.1"
PORT=9104
PID_FILE="paddle_ocr_server.pid"

# Optional: activate env
# source /share/data/drive_4/haider/miniconda3/bin/activate paddleocr_srv

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "PaddleOCR server already running (PID $(cat $PID_FILE))"
        exit 0
    fi

    echo "Starting PaddleOCR server on ${HOST}:${PORT}"

    nohup uvicorn ${APP} \
        --host ${HOST} \
        --port ${PORT} \
        --workers 1 \
        > paddle_ocr_server.log 2>&1 &

    echo $! > "$PID_FILE"
    sleep 2

    if kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "PaddleOCR server started (PID $(cat $PID_FILE))"
    else
        echo "Failed to start PaddleOCR server"
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "PaddleOCR server not running (no PID file)"
        exit 0
    fi

    PID=$(cat "$PID_FILE")

    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping PaddleOCR server (PID $PID)"
        kill "$PID"
        sleep 3

        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing PaddleOCR server"
            kill -9 "$PID"
        fi
    else
        echo "Stale PID file found"
    fi

    rm -f "$PID_FILE"
    echo "PaddleOCR server stopped"
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "PaddleOCR server running (PID $(cat $PID_FILE))"
    else
        echo "PaddleOCR server not running"
    fi
}

restart() {
    stop
    start
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
