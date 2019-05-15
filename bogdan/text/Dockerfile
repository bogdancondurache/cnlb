FROM python:3.7.3-alpine

COPY . /app
WORKDIR /app
RUN apk --no-cache add libstdc++ \
    && apk --no-cache add --virtual build-dependencies g++ musl-dev \
    && pip3 install -r requirements.txt
ENV FLASK_APP web.py
ENTRYPOINT ["python3"]
CMD ["-m", "flask", "run"]
