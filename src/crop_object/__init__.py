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
    if len(img.shape) > 2:
        contours, _ = cv2.findContours(img[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find the largest contour (i.e. the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the bounding box of the object
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the object
        xx_coordinate = x + w
        yy_coordinate = y + h
        cropped_img = img[y:yy_coordinate, x:xx_coordinate]

        info = "crop_object\t"
        info += f"x_coordinate={x}\t"
        info += f"y_coordinate={y}\t"
        info += f"xx_coordinate={xx_coordinate}\t"
        info += f"yy_coordinate={yy_coordinate}"

        return cropped_img, info
    except ValueError:
        return None, None

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

