import base64
from flask import Flask, request, send_from_directory
import os
import json
from pathlib import Path
import secrets
import uuid
import time
from datetime import date
import glob
from PIL import Image
from numpy import asarray
from pycivitai import civitai_download

import modules.async_worker as worker
import modules.advanced_parameters as advanced_parameters

app = Flask(__name__, static_url_path='/')
app.secret_key = secrets.token_urlsafe(32)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['CAPTURE_FOLDER'] = './captures/'

@app.route('/api/processing', methods=['POST'])
def process_image():
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    advanced_parameters.set_all_advanced_parameters(
        False,
        1.5,
        0.8,
        0.3,
        7,
        'dpmpp_2m_sde_gpu',
        'karras',
        False,
        -1, -1, -1, -1, -1, -1,
        False, False, False, False,
        0.25, 64, 128, 'joint', False,
        1.01, 1.02, 0.99, 0.95, False, False, 'v2.6', 1, 0.618
    )

    filename = f"%s.jpg" % uuid.uuid4()
    filepath = Path(app.config['CAPTURE_FOLDER'], filename)

    image_file = open(filepath, "wb")
    image_file.write(base64.b64decode(request.form.get('file')))
    image_file.close()

    output_folder = Path("outputs", str(date.today()))

    img = Image.open(filepath)
    numpydata = asarray(img)

    prompt = request.form.get('prompt', '')
    negative_prompt = request.form.get('negative_prompt', '')
    loras = json.loads(request.form.get('loras', []))

    for lora in loras:
        lora_path = civitai_download(
            lora["civitai_id"],
            version=lora["civitai_version"]
        )

    args = [
        prompt, negative_prompt,
        ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Negative', 'Fooocus Photograph'],
        'Speed',
        '896×1152 <span style="color: grey;"> ∣ 7:9</span>',
        1,
        '4993914057145546898',
        2,
        4,
        'sd_xl_base_1.0_0.9vae.safetensors',
        'sd_xl_refiner_1.0_0.9vae.safetensors',
        0.8,
        'sd_xl_offset_example-lora_1.0.safetensors', 0.1,
        'None', 1,
        'None', 1,
        'None', 1,
        'None', 1,
        True,
        'ip',
        'Disabled',
        None,
        [],
        None,
        '',
        numpydata, 0.5, 1, 'FaceToPrompt',
        numpydata, 1, 1.1, 'PyraCanny',
        numpydata, 0.5, 1, 'InsightFaceSwap',
        None, 0.5, 0.6, 'ImagePrompt'
    ]

    task = worker.AsyncTask(args=list(args))
    worker.async_tasks.append(task)
    finished = False

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

            if flag == 'finish':
                finished = True

    result_filename = Path(max(output_folder.glob('*.png'), key=os.path.getctime)).name
    print(f'Result filename: %s' % result_filename)

    return send_from_directory(
        output_folder,
        result_filename,
        as_attachment=True
    )


port = int(os.environ.get('PORT', 5005))
app.run(host='0.0.0.0', port=port, use_reloader=False)
