FROM pnelsen129/fashion-mnist:2

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
