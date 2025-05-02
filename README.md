The console will display the recognized digit from 0 to 9.

## Input image format

- The image will be automatically resized to 28x28 pixels
- The image will be converted to grayscale
- PNG, JPEG and other formats supported by the Pillow library are supported

## Project structure

- `mnist.py` - the main script for training and prediction
- `number.png` - an example image with a handwritten digit for testing (not included in the repository)

## Running the project
- python mnist.py --train # for training
- python mnist.py --predict number.png # for prediction

## License

The project is open for use and modification.