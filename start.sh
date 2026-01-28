#!/bin/bash
set -e

echo "Starting RAG Application..."

# Create necessary directories
mkdir -p uploads data

# Wait for Redis to be ready (simplified)
if [ -n "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    python3 -c "
import redis
import sys
import time
from urllib.parse import urlparse

url = '$REDIS_URL'
parsed = urlparse(url)
host = parsed.hostname or 'redis'
port = parsed.port or 6379

for i in range(30):
    try:
        r = redis.Redis(host=host, port=port, socket_connect_timeout=2)
        r.ping()
        print('Redis is ready!')
        sys.exit(0)
    except Exception as e:
        if i < 29:
            time.sleep(1)
        else:
            print(f'Redis not available: {e}')
            sys.exit(1)
"
fi

# Start the application
echo "Starting API server..."
exec gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    -w "${WORKERS:-2}" \
    -b 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --log-level "${LOG_LEVEL:-info}" \
    --timeout "${TIMEOUT:-120}"
