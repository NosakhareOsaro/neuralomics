FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY common/ common/
COPY modules/1_cellpainting/ modules/1_cellpainting/

RUN pip install --no-cache-dir -e ".[cellpainting]"

# No training/eval entrypoint script exists yet (module is still a
# scaffold). Invoke explicitly once one lands, e.g.:
#   docker run <image> python -m cellpainting.train --config configs/train.yaml
CMD ["python", "-c", "import cellpainting; print(cellpainting.__doc__)"]
