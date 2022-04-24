FROM pnelsen129/fashion-mnist:1

WORKDIR /app

RUN pip install flask tensorflow Pillow

COPY backend .

CMD ["python3", "app.py"]
