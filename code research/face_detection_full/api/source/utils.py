import os
import cv2 # OpenCV for image editing, computer vision and deep learning
import base64 # Used for encoding image content string
import numpy as np # Numpy for math/array operations
from matplotlib import pyplot as plt # Matplotlib for visualization
import face_recognition

def draw_rectangle(image, face, file, index, code):
    (start_x, start_y, end_x, end_y) = face["rect"]
    # Arrange color of the detection rectangle to be drawn over image
    detection_rect_color_rgb = (0, 255, 255)
    # Draw the detection rectangle over image
    cv2.rectangle(img = image, 
                  pt1 = (start_x, start_y), 
                  pt2 = (end_x, end_y), 
                  color = detection_rect_color_rgb, 
                  thickness = 5)
    is_target = False
    if (file !=None):
        is_target = match_face(file, index, code)

    # Draw detection probability, if it is present
    if face["prob"] != []:
        # Create probability text to be drawn over image
        text = "{:.2f}%".format(face["prob"])
        if(file!=None):
            text = "Unknown {:.2f}%".format(face["prob"])
            if(is_target):
                text = "Match {:.2f}%".format(face["prob"])
       
        # Arrange location of the probability text to be drawn over image
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        # Arrange color of the probability text to be drawn over image
        probability_color_rgb = (0, 255, 255)
        # Draw the probability text over image
        cv2.putText(img = image, 
                    text = text, 
                    org = (start_x, y), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1.2, 
                    color = probability_color_rgb, 
                    thickness = 3)

def draw_rectangles(image, faces, file, code=[]):
    # Draw rectangle over detections, if any face is detected
    if len(faces) == 0:
        num_faces = 0
    else:
        num_faces = len(faces)
        # Draw a rectangle
        index = 0
        faces.reverse()
        for face in faces:
            draw_rectangle(image, face, file, index, code)
            index +=1
    return num_faces, image

def read_image(file):
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    return image


def prepare_image(image):
    # Create string encoding of the image
    image_content = cv2.imencode('.jpg', image)[1].tostring()
    # Create base64 encoding of the string encoded image
    encoded_image = base64.encodestring(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return to_send


def plot_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def get_folder_dir(folder_name):
    cur_dir = os.getcwd()
    folder_dir = cur_dir + "/" + folder_name + "/"
    return folder_dir


def match_face(file, index, code):
    try:
        # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)
        #known_face_encoding = [-0.09634063,  0.12095481, -0.00436332, -0.07643753,  0.0080383,
                            #     0.01902981, -0.07184699, -0.09383309,  0.18518871, -0.09588896,
                            #     0.23951106,  0.0986533 , -0.22114635, -0.1363683 ,  0.04405268,
                            #     0.11574756, -0.19899382, -0.09597053, -0.11969153, -0.12277931,
                            #     0.03416885, -0.00267565,  0.09203379,  0.04713435, -0.12731361,
                            # -0.35371891, -0.0503444 , -0.17841317, -0.00310897, -0.09844551,
                            # -0.06910533, -0.00503746, -0.18466514, -0.09851682,  0.02903969,
                            # -0.02174894,  0.02261871,  0.0032102 ,  0.20312519,  0.02999607,
                            # -0.11646006,  0.09432904,  0.02774341,  0.22102901,  0.26725179,
                            #     0.06896867, -0.00490024, -0.09441824,  0.11115381, -0.22592428,
                            #     0.06230862,  0.16559327,  0.06232892,  0.03458837,  0.09459756,
                            # -0.18777156,  0.00654241,  0.08582542, -0.13578284,  0.0150229 ,
                            #     0.00670836, -0.08195844, -0.04346499,  0.03347827,  0.20310158,
                            #     0.09987706, -0.12370517, -0.06683611,  0.12704916, -0.02160804,
                            #     0.00984683,  0.00766284, -0.18980607, -0.19641446, -0.22800779,
                            #     0.09010898,  0.39178532,  0.18818057, -0.20875394,  0.03097027,
                            # -0.21300618,  0.02532415,  0.07938635,  0.01000703, -0.07719778,
                            # -0.12651891, -0.04318593,  0.06219772,  0.09163868,  0.05039065,
                            # -0.04922386,  0.21839413, -0.02394437,  0.06173781,  0.0292527 ,
                            #     0.06160797, -0.15553983, -0.02440624, -0.17509389, -0.0630486 ,
                            #     0.01428208, -0.03637431,  0.03971229,  0.13983178, -0.23006812,
                            #     0.04999552,  0.0108454 , -0.03970895,  0.02501768,  0.08157793,
                            # -0.03224047, -0.04502571,  0.0556995 , -0.24374914,  0.25514284,
                            #     0.24795187,  0.04060191,  0.17597422,  0.07966681,  0.01920104,
                            # -0.01194376, -0.02300822, -0.17204897, -0.0596558 ,  0.05307484,
                            #     0.07417042,  0.07126575,  0.00209804]       
        known_face_encoding = code
        img = face_recognition.load_image_file(file)

        # Get face encodings for any faces in the uploaded image
        unknown_face_encodings = face_recognition.face_encodings(img)

        is_target = False

        if len(unknown_face_encodings) > 0:
            # See if the first face in the uploaded image matches the known face of Obama
            match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings[index])
            if match_results[0]:
                is_target = True
    except:
        is_target = False
    
    return is_target


def set_face_code(file):
    result = []
    img = face_recognition.load_image_file(file)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)
    if len(unknown_face_encodings) > 0:
        result = unknown_face_encodings[0]
    
    return result
