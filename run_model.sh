#!/bin/bash
# Rebuild the image with no cache, then run prediction
# if changes are made run this line as well 
# docker build --no-cache -t pymc-mmm:latest .
docker run --rm -v "$(pwd)/artifacts":/app/artifacts -v "$(pwd)/input":/data/input -v "$(pwd)/output":/data/output pymc-mmm:latest python -m app.predict_cli --new_csv /data/input/new_media.csv --out_csv /data/output/predictions.csv