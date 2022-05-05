FROM pnelsen129/fashion-mnist

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
