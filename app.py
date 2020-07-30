from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)

def read_b64_as_bw_cv(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2GRAY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webFaceID', methods=["POST"])
def process_image():
    payload = request.form.to_dict(flat=False)
    im_b64 = payload['imgBase64'][0]
    im_b64_code = im_b64[im_b64.find(',')+1:]
    im_binary = base64.b64decode(im_b64_code)
    buf = io.BytesIO(im_binary)
    pilImage = Image.open(buf)

    # face detection & bounding box extraction
    cvGreyImage = cv2.cvtColor(np.array(pilImage), cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        cvGreyImage,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    for index, (x, y, w, h) in enumerate(faces):
        ext_face = cv2.resize(cvGreyImage[y : y+h, x : x+w],
                              (100,115), 
                              interpolation = cv2.INTER_AREA)
        cv2.imwrite('face'+str(index)+'.jpg', ext_face)
    
    print("Found {0} Faces!".format(len(faces)))

    # face recognition


    return jsonify({'msg': 'success'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')