from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image


app = Flask(__name__)

@app.route('/',methods=['post','get']) # will use get for the first page-load, post for the form-submit
def predict(): # this function can have any name
    try:
        model = load_model('yogamodel.h5') # the mymodel.h5 file was created in Colab, downloaded and uploaded using Filezilla
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            test_image = image.load_img(uploaded_file, target_size=[28, 28], color_mode='grayscale')
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            result = model.predict(test_image/255.0)
            result = result[0].argmax()
            switcher = {0:"Downward Dog", 1:"Head Stand", 2:"Crow", 3:"Chaturanga"}
            string_result = switcher.get(result)

            return render_template('index.html', result=string_result)
        else:
            return render_template('index.html', result='No input(s)')
            # calling render_template will inject the variable 'result' and send index.html to the browser
    except Exception as e:
        return render_template('index.html', result= str(e))

if __name__ == '__main__':
    app.run()
