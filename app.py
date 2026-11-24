import os
from functools import cache
from pathlib import Path
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import BooleanField, FileField, SubmitField, FloatField, HiddenField, IntegerField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import APP_DECODER, VGGEncoder, Decoder
from utils.utils import style_transfer


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    style_percent = IntegerField('Style strength', default=80)
    face_percent = IntegerField('Facial detail preservation', default=70)
    preserve_detail = BooleanField('Preserve face detail')
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vgg_path = os.path.join(BASE_DIR, 'vgg_normalised.pth')

encoder = VGGEncoder(vgg_path).to(device)
decoder = Decoder().to(device)
decoder.load_weights(APP_DECODER)

encoder.eval()
decoder.eval()

@cache
def get_portrait_decoder():
    path = Path(BASE_DIR) / 'experiment' / 'pilot_600' / 'decoder_step_000200.pth'
    model = Decoder()
    model.load_weights(path)
    return model.to(device).eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)



@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')

                portrait = form.preserve_detail.data
                manual = 'style_percent' in request.form or 'face_percent' in request.form
                if portrait:
                    strength, detail = 80, 70
                    prefix = 'portrait_style80_'
                elif manual:
                    strength, detail = form.style_percent.data, form.face_percent.data
                    if not all(v is not None and 0 <= v <= 100 for v in (strength, detail)):
                        raise ValueError("Choose percentages from 0 to 100.")
                    prefix = f'style{strength}_face{detail}_'
                if portrait or manual:
                    stylized_image = style_transfer(content_image, style_image, encoder, get_portrait_decoder(),
                                                    strength / 100, device, preserve_detail=True,
                                                    face_detail=detail / 100)
                else:
                    # Older clients submit a single normalized alpha.
                    prefix = 'stylized_'
                    stylized_image = style_transfer(content_image, style_image, encoder, decoder, form.alpha.data, device)
                result_filename = prefix + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                
                result_image = result_filename
            except Exception as e:
                error = str(e)
    elif request.method == 'POST':
        if not content_filename:
            error = 'Please upload content image'
        if not style_filename:
            error = 'Please upload style image'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)






