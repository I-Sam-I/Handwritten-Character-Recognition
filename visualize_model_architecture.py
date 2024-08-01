import visualkeras
from PIL import ImageFont
from tensorflow.keras.models import load_model
import os


DIRECTORY = 'Model Architectures'

def main():
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path)

    digit_model = load_model('models/Digit_Model.keras')
    uppercase_model = load_model('models/Uppercase_Model.keras')
    lowercase_model = load_model('models/Lowercase_Model.keras')
    classifier_model = load_model('models/Classifier_Model.keras')

    # Ensure the directory exists
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    visualkeras.layered_view(digit_model, legend=True, to_file=f'{DIRECTORY}/digit_model.png', font=font)
    visualkeras.layered_view(uppercase_model, legend=True, to_file=f'{DIRECTORY}/uppercase_model.png', font=font)
    visualkeras.layered_view(lowercase_model, legend=True, to_file=f'{DIRECTORY}/lowercase_model.png', font=font)
    visualkeras.layered_view(classifier_model, legend=True, to_file=f'{DIRECTORY}/classifier_model.png', font=font)



if __name__ == '__main__':
    main()