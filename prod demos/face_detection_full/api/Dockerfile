FROM python:3.6
#update
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y



#install requirements
COPY ./requirements.txt /tmp/requirements.txt
WORKDIR /tmp
RUN pip3 install cmake
RUN pip3 install -r requirements.txt

#copy app
COPY . /api
WORKDIR /


CMD ["gunicorn", "-w", "3", "-b", ":5000", "-t", "360", "--reload", "api.app:app"]