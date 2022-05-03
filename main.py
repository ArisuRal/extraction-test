import io
from flask import Flask
from flask import render_template, request
#from flask_ngrok import run_with_ngrok

import cv2
import numpy as np
from swtloc import SWTLocalizer
import pytesseract

app = Flask(__name__, template_folder='template')
#run_with_ngrok(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


@app.route('/', methods=['GET'])
def home():
    return render_template(
        '/upload_form.html',
        phrase_output="",
        isSubmit=False
    )


@app.route('/', methods=['POST'])
def do_something():
    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']  # the posted image will be stored here
        in_memory = io.BytesIO()
        photo.save(in_memory)

        data = np.fromstring(in_memory.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Resize image
        scale_percent = 80
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # Text extraction process
        swt = SWTLocalizer()

        if 'dark-text' in request.form:
            mode = 'lb_df'
        else:
            mode = 'db_lf'

        swt.swttransform(image=resized, edge_func='ac', ac_sigma=1.0, text_mode=mode,
                             gs_blurr=True, blurr_kernel=(5, 5), minrsw=3,
                             maxCC_comppx=5000, maxrsw=200, max_angledev=np.pi / 6,
                             acceptCC_aspectratio=5.0)

        # Text identification (OCR)
        result = pytesseract.image_to_string(swt.swt_mat, lang='eng', config='--psm 4')

        return render_template(
            '/upload_form.html',
            phrase_output=result.strip(),
            isSubmit=True
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

    # host="0.0.0.0", port=5000, debug=True
    # host='0.0.0.0' means this will be hosted in your IP address instead of local host
    # in the cmd prompt type in ipconfig to get your IPv4

