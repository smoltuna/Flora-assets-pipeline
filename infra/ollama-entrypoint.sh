#!/bin/bash
# Start ollama serve in background
ollama serve &

# Wait for server to be ready
until ollama list > /dev/null 2>&1; do
    sleep 1
done

echo "Ollama ready — pulling models..."

ollama pull nomic-embed-text
echo "nomic-embed-text ready."

ollama pull llama3.2:3b
echo "llama3.2:3b ready."

echo "All models pulled. Ollama entrypoint complete."

# Keep container running
wait
