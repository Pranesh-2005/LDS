FROM python:3.10.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/

# Ensure upload folder exists

EXPOSE 5000

CMD ["python", "backend/app.py"]