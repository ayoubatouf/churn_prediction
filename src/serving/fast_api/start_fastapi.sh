#!/bin/bash

echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
SERVER_PID=$!  # Capture the server's process ID

echo "Waiting for the FastAPI server to start..."

URL="http://localhost:8000/api/predict/"

while ! curl -X OPTIONS -s "$URL" > /dev/null; do
    echo "Waiting for server..."
    sleep 1
done

echo "Server is up and running on $URL."
echo "Server PID: $SERVER_PID"
