'''Crop an object in a black and white image.'''

import cv2
import numpy as np

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except:
    IN_COLAB = False
logger.info(f"Running in Google Colab: {IN_COLAB}")

def crop_largest_object(img):
    # Find contours of the object
    contours, _ = cv2.findContours(img[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (i.e. the object)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box of the object
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the object
    cropped_img = img[y:y+h, x:x+w]

    info = f"crop_object\t{x} {y} {w} {h} # x y width height"

    return cropped_img, info

if __name__ == "__main__":

    def int_or_str(text):
        '''Helper function for argument parsing.'''
        try:
            return int(text)
        except ValueError:
            return text

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.description = __doc__

    parser.add_argument("-i", "--input", type=int_or_str,
                        help="Prefix of the input image sequence",
                        default="/tmp/input")
    
    parser.add_argument("-o", "--output", type=int_or_str,
                        help="Prefix of the output image sequence",
                        default="/tmp/output")

    parser.add_argument("-e", "--extension", type=int_or_str,
                        help="Image extension",
                        default=".png")

    args = parser.parse_args()

    import sequence_iterator

    class CropSequence(sequence_iterator.ImageSequenceIterator):
        def process(self, image):
            return crop_largest_object(image)

    iterator = WarpSequence(
        input_sequence_prefix="/tmp/input",
        output_sequence_prefix="/tmp/output",
        image_extension="png")

    iterator.process_sequence()

    logger.info(f"Your files should be in {args.output}")

