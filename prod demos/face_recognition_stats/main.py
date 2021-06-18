from werkzeug.utils import secure_filename

from face_recognition import api as face_recognition
import glob
import os
import click
import multiprocessing
import itertools
import re
import PIL.Image
import numpy as np
import shutil
from fastapi import FastAPI, Form, File, UploadFile, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
KNOWN_IMAGES_LOCATION = './known_images'
TEST_IMAGES_LOCATION = './test_images'
recognition_results = {}

app = FastAPI()

app.mount("/demo", StaticFiles(directory="static", html = True), name="static")

@app.get("/known_images/{filename}")
async def read_known_image(filename):
    file_location = f"{KNOWN_IMAGES_LOCATION}/{filename}"
    return FileResponse(file_location)

@app.get("/test_images/{filename}")
async def read_unknown_image(filename):
    file_location = f"{TEST_IMAGES_LOCATION}/{filename}"
    return FileResponse(file_location)

@app.post('/process')
async def compare_faces(test_image: UploadFile = File(...), known_image: UploadFile = File(...), tolerance: Optional[float] = Form(0.6)):
    clear_directory(TEST_IMAGES_LOCATION)
    clear_directory(KNOWN_IMAGES_LOCATION)
    recognition_results.clear()
    test_upload = test_image.file
    known_upload = known_image.file
    print('Tolerance value is ' + str(tolerance))
    test_filename = secure_filename(test_image.filename)
    known_filename = secure_filename(known_image.filename)

    test_file_location = f"{TEST_IMAGES_LOCATION}/{test_filename}"
    with open(test_file_location, "wb+") as file_object:
        shutil.copyfileobj(test_upload, file_object) 

    known_file_location = f"{KNOWN_IMAGES_LOCATION}/{known_filename}"
    with open(known_file_location, "wb+") as file_object:
        shutil.copyfileobj(known_upload, file_object)

    do_detect(KNOWN_IMAGES_LOCATION, test_file_location, tolerance)
    confidence = recognition_results[0][1]*100
    data = positive_match(test_file_location, known_file_location, confidence, tolerance)
    if (isinstance(confidence, str)):
        data = negative_match(test_file_location, known_file_location, tolerance)

    return Response(content=data, media_type="text/html")

def positive_match(test_file_location, known_file_location, confidence, tolerance):
    data = ''' \
    <!doctype html> \
    <title>Results</title> \
    <img height=420 src={unknown}/>
    <img height=420 src={known}/>
    <p>Confidence that this is the same person: {confidence}%</p> \
    <p>This comparison used a tolerance value of {tolerance}</p>
    <a href='/demo'>Try Again</a> \
    '''.format(unknown=test_file_location, known=known_file_location, confidence=confidence, tolerance=tolerance)
    return data

def negative_match(test_file_location, known_file_location, tolerance):
    data = ''' \
    <!doctype html> \
    <title>No Results</title> \
    <img height=420 src={unknown}/>
    <img height=420 src={known}/>
    <p>Unable to determine a facial recognition match using a tolerance value of {tolerance}</p> \
    <p>Try again with a different photo or try adjusting the tolerance.</p>
    <a href='/demo'>Try Again</a> \
    '''.format(unknown=test_file_location, known=known_file_location, tolerance=tolerance)
    return data

def clear_directory(dir):
    files = glob.glob(f"{dir}/*")
    for f in files:
        os.remove(f)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings

def print_result(filename, name, distance):
    if distance is None:
        print("{},{}".format(filename, name))
        recognition_results[0] = name
        return
    print("{},{},{}".format(filename, name, distance))
    recognition_results[0] = name, 1-distance

def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)

        if True in result:
            [print_result(image_to_check, name, distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
        else:
            print_result(image_to_check, "unknown_person", None)

    if not unknown_encodings:
        # print out fact that no faces were found in image
        print_result(image_to_check, "no_persons_found", None)

def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )

    pool.starmap(test_image, function_parameters)

def do_detect(known_people_folder, image_to_check, tolerance=0.6):
    known_names, known_face_encodings = scan_known_people(known_people_folder)

    test_image(image_to_check, known_names, known_face_encodings, tolerance)