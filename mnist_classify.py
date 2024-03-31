from tensorflow.keras.models import load_model
MODEL_NAME =   'model_fmr_all.keras'
import numpy as np
from PIL import Image, ImageOps
model = load_model(MODEL_NAME)
INPUT_SHAPE = (28, 28, 1)


DEBUG_PRINTS = False


def worker(image):
    # Open and preprocess image
    if isinstance(image, str):
        image = Image.open(image)
    image = ImageOps.grayscale(image)
    image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))

    # Convert to numpy array for input into model
    model_input = np.array(image, dtype="float32")
    model_input = np.reshape(model_input,(1,INPUT_SHAPE[1],INPUT_SHAPE[0],1))

    # Normalize input data
    coldest_pixel = float(model_input.min())
    hottest_pixel = float(model_input.max())

    # Make it moar contrast
    model_input = ((model_input - coldest_pixel) / max(1.,(hottest_pixel-coldest_pixel)) - 0.5) * 2 + 0.5
    model_input[np.where(model_input < 0.)] = 0.
    model_input[np.where(model_input > 1.)] = 1.

    # Invert if background is white (determined by a pixel in a corner)
    if model_input[0,0,0,0] > 0.5:
        model_input = 1. - model_input

    # Train data weren't normalized so return data to the scale [0. .. 255.]
    model_input = model_input * 255.

    if DEBUG_PRINTS:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt_data = model_input[0].squeeze()
        plt.imshow(plt_data, cmap='gray')
        plt.show()

    # Predict class and return result
    prediction = model.predict(model_input)
    highest = np.argmax(prediction[0])
    return model_input[0], highest, prediction[0][highest], prediction[0]


if __name__ == "__main__":
    for file_path in (
        "./test_data/digit-9.png",
        "./test_data/digit-2a.png",
        "./test_data/digit-2b.png",
        "./test_data/digit-4.png",
    ):
        result = worker(file_path)
        print(f"{file_path}: {result}")
