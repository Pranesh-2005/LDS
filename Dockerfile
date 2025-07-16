FROM python:3.10.13-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/
COPY backend/leaf_model.onnx ./backend/leaf_model.onnx

# Ensure upload folder exists

EXPOSE 5000

CMD ["python", "backend/app.py"]