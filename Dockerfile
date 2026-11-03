FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements-prod.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "predict:app"]