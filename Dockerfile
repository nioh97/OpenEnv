FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY openenv_farm /app/openenv_farm
COPY inference.py /app/inference.py

RUN pip install --no-cache-dir -r /app/openenv_farm/requirements.txt

CMD ["uvicorn", "openenv_farm.api.server:app", "--host", "0.0.0.0", "--port", "7860"]
