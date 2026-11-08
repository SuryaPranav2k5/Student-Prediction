FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime dependencies for the FastAPI service (model + SHAP explanation only).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pandas numpy scikit-learn lightgbm shap pyarrow joblib fastapi uvicorn

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
