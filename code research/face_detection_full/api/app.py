import os
from flask import Flask,jsonify,request,render_template
from .source.face_detection import detect_faces_with_ssd
from .source.utils import draw_rectangles, read_image, prepare_image, set_face_code
from .config import DETECTION_THRESHOLD
global Encoding 
global target_image 
Encoding = [-0.09634063,  0.12095481, -0.00436332, -0.07643753,  0.0080383,
                                0.01902981, -0.07184699, -0.09383309,  0.18518871, -0.09588896,
                                0.23951106,  0.0986533 , -0.22114635, -0.1363683 ,  0.04405268,
                                0.11574756, -0.19899382, -0.09597053, -0.11969153, -0.12277931,
                                0.03416885, -0.00267565,  0.09203379,  0.04713435, -0.12731361,
                            -0.35371891, -0.0503444 , -0.17841317, -0.00310897, -0.09844551,
                            -0.06910533, -0.00503746, -0.18466514, -0.09851682,  0.02903969,
                            -0.02174894,  0.02261871,  0.0032102 ,  0.20312519,  0.02999607,
                            -0.11646006,  0.09432904,  0.02774341,  0.22102901,  0.26725179,
                                0.06896867, -0.00490024, -0.09441824,  0.11115381, -0.22592428,
                                0.06230862,  0.16559327,  0.06232892,  0.03458837,  0.09459756,
                            -0.18777156,  0.00654241,  0.08582542, -0.13578284,  0.0150229 ,
                                0.00670836, -0.08195844, -0.04346499,  0.03347827,  0.20310158,
                                0.09987706, -0.12370517, -0.06683611,  0.12704916, -0.02160804,
                                0.00984683,  0.00766284, -0.18980607, -0.19641446, -0.22800779,
                                0.09010898,  0.39178532,  0.18818057, -0.20875394,  0.03097027,
                            -0.21300618,  0.02532415,  0.07938635,  0.01000703, -0.07719778,
                            -0.12651891, -0.04318593,  0.06219772,  0.09163868,  0.05039065,
                            -0.04922386,  0.21839413, -0.02394437,  0.06173781,  0.0292527 ,
                                0.06160797, -0.15553983, -0.02440624, -0.17509389, -0.0630486 ,
                                0.01428208, -0.03637431,  0.03971229,  0.13983178, -0.23006812,
                                0.04999552,  0.0108454 , -0.03970895,  0.02501768,  0.08157793,
                            -0.03224047, -0.04502571,  0.0556995 , -0.24374914,  0.25514284,
                                0.24795187,  0.04060191,  0.17597422,  0.07966681,  0.01920104,
                            -0.01194376, -0.02300822, -0.17204897, -0.0596558 ,  0.05307484,
                                0.07417042,  0.07126575,  0.00209804]
target_image = 'obama.jpeg'


app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
  return render_template('index.html',target_image=target_image)

@app.route('/detectz')
def detectz():
  return render_template('detect.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']

    # Read image
    image = read_image(file)
    
    # Detect faces
    faces = detect_faces_with_ssd(image, min_confidence=DETECTION_THRESHOLD)

    return jsonify(detections = faces)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']

    # Read image
    image = read_image(file)
    
    # Detect faces
    faces = detect_faces_with_ssd(image, DETECTION_THRESHOLD)
    
    # Draw detection rects
    num_faces, image = draw_rectangles(image, faces, file, Encoding)
    
    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('index.html',target_image=target_image, face_detected=len(faces)>0, num_faces=len(faces), image_to_show=to_send, init=True)

@app.route('/upload_code', methods=['POST'])
def upload_code():
    #save the file so it can be seen 
    f = request.files['image']
    global target_image
    target_image =f.filename
    if f.filename != '':
            f.save('api/static/images/'+f.filename)

    file = request.files['image']
    # Detect faces
    global Encoding 
    Encoding= set_face_code(file)

    return render_template('index.html',target_image=target_image)


@app.route('/upload2', methods=['POST'])
def upload2():
    file = request.files['image']

    # Read image
    image = read_image(file)
    
    # Detect faces
    faces = detect_faces_with_ssd(image, DETECTION_THRESHOLD)
    
    # Draw detection rects
    num_faces, image = draw_rectangles(image, faces, None)
    
    # Prepare image for html
    to_send = prepare_image(image)

    return render_template('detect.html', face_detected=len(faces)>0, num_faces=len(faces), image_to_show=to_send, init=True)



if __name__ == '__main__':
    app.run(debug=True, 
            use_reloader=True,
            port=3000)
