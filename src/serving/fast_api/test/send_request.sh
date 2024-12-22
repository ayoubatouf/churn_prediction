#!/bin/bash

URL="http://localhost:8000/api/predict/"

# Check if the input file exists
if [[ ! -f "fast_api_input.json" ]]; then
    echo "Error: fast_api_input.json file not found!"
    exit 1
fi

echo "Sending POST request to $URL"
curl -X POST "$URL" -H "Content-Type: application/json" -d @fast_api_input.json

echo "Request sent."
