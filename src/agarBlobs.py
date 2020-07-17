import time
import os
import cv2
import math
import numpy as np
from src import quickGrab, settings

PI = math.pi

def load_last_image(path):
    images = []

    for file in os.listdir(path):
        if file.endswith('.png'):
            images.append(file)
    last_image = cv2.imread(path + '\\' + images[len(images) - 1])
    return last_image


def show(img, title="Image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)


def blur(img, rad=settings.BLUR_RAD):
    return cv2.blur(img, (rad, rad))


def grey(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def remove_ui(img):
    # Create lookup table
    lut = []
    for i in range(256):
        lut.append(255 if (i == 174 or i == 149 or i == 195) else i)

    lut = np.array(lut, dtype=np.uint8)

    return lut[img]


def get_spikes(img):
    lut = []
    for i in range(256):
        lut.append(0 if i == settings.SPIKES_COLOR else 255)

    lut = np.array(lut, dtype=np.uint8)
    return lut[img]


def binary(img, thresh=settings.BINARY_THRESHOLD):
    (_, img_bin) = cv2.threshold(img, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
    return img_bin


def canny(img):
    return cv2.Canny(img, 0, 5, 100)


def preprocess(img):
    img_grey = grey(img)

    img_spikes = get_spikes(img_grey)
    img_spikes_blur = blur(img_spikes, settings.SPIKES_BLUR_RAD)
    img_spikes_bin = binary(img_spikes_blur, settings.SPIKES_BINARY_THRESHOLD)

    img_rem = remove_ui(img_grey)
    img_blur = blur(img_rem, settings.BLUR_RAD)
    img_bin = binary(img_blur, settings.BINARY_THRESHOLD)

    if settings.DEBUG:
        show(img_rem)
        # show(img_spikes_blur)
        # show(img_spikes_bin)
        show(img_blur)
        show(img_bin)

    return img_bin, img_spikes_bin


def get_contours(img):
    img_edged = canny(img)
    contours, _ = cv2.findContours(img_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def check_concavity(img, contours):
    for cont in contours:
        hull = cv2.convexHull(cont, returnPoints=False)
        defects = cv2.convexityDefects(cont, hull)
        print(defects)
        print("--")

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def update_bbox(blob, bbox):
    x, y, size, r = blob

    if x - r < bbox[0]:
        bbox[0] = x - r
    if x + r > bbox[2]:
        bbox[2] = x + r
    if y - r < bbox[1]:
        bbox[1] = y - r
    if y + r > bbox[3]:
        bbox[3] = y + r


def compare_spikes(blob, blobs_spikes):
    for spike in blobs_spikes:
        delta = dist(spike, blob)
        if delta < 75:
            return True

    return False


def hide_spikes(img, spikes):
    for spike in spikes:
        (x, y, size, r) = spike
        cv2.circle(img, (x, y), int(r * 1.1), (255, 255, 255), cv2.FILLED)  # Draw white circle over spike


def check_game_over(img):
    hist = np.bincount(img.ravel(), minlength=256)
    main_color = list(hist).index(hist.max())
    print("Main color:", main_color)
    return main_color == settings.END_SCREEN_COLOR

def get_blob_from_contour(contour):
    """Higher level function"""
    (x0, y0), r0 = cv2.minEnclosingCircle(contour)
    x = int(x0)
    y = int(y0)
    r = int(r0)

    # Compute extent
    cont_area = cv2.contourArea(contour)
    circle_area = PI * r ** 2
    if cont_area == 0:
        return None
    extent = float(circle_area / cont_area)

    # Use the exact area, unless sticking out of screen, then use approx
    size = int(cont_area) if extent < 2 else int(circle_area)
    blob = (x, y, size, r)

    return blob


def draw_blobs(img, blobs, my_blob):
    for (i, blob) in enumerate(blobs):
        (x, y, size, r) = blob
        rel_size = size / my_blob[2]
        if settings.DEBUG:
            print("Contour #%d" % i)
            print("\tSize: %d" % size)
        if img.any():
            smaller = rel_size < 1
            color = (0, 0, 255) if not smaller else (0, 255, 255)
            if not smaller or settings.DEBUG:
                cv2.circle(img, (x, y), int(r), color, 3)
                cv2.putText(img, str(i), ((x + r), y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def find_blobs(contours, blobs_spikes=[], img=[]):
    """Lower level function"""
    blobs = []
    closest_to_center = 0
    closest_dist = 9999
    insert_ind = 0
    blobs_bbox = [1000, 500, 1001, 501]

    for (i, c) in enumerate(contours):
        blob = get_blob_from_contour(c)  # (x, y, size, r)
        if not blob:
            continue
        (x, y, size, r) = blob

        # Update blob bbox
        update_bbox(blob, blobs_bbox)

        # Check if spikes
        is_spikes = compare_spikes(blob, blobs_spikes)

        if not is_spikes:
            # Check if player blob
            dist_center = dist((x, y), settings.SCREEN_CENTER)
            if dist_center < closest_dist and closest_to_center > -1:
                closest_dist = dist_center
                closest_to_center = insert_ind

            blobs.append(blob)
            insert_ind += 1

    # Determine player blob
    my_blob = None
    if len(blobs) > 0:
        my_blob = blobs[closest_to_center]
        blobs.remove(my_blob)
        if settings.DEBUG or settings.SAVE_PROCESSED:
            (mx, my, msize, mr) = my_blob
            cv2.circle(img, (mx, my), int(mr), (255, 0, 0), 3)
            cv2.putText(img, "P", ((mx + mr), my), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if settings.DEBUG:
        draw_blobs(img, blobs, my_blob)

    return blobs, my_blob, blobs_bbox


def find_spikes(contours_spikes, img=None):
    """Lower level function"""
    # Analyse spikes
    blobs_spikes = []
    for (i, c) in enumerate(contours_spikes):
        spike = get_blob_from_contour(c)  # (x, y, size, r)
        if not spike:
            continue
        (x, y, size, r) = spike

        r = int(math.sqrt(size / math.pi))
        blobs_spikes.append(spike)

        if settings.DEBUG or settings.SAVE_PROCESSED:
            cv2.circle(img, (x, y), int(r), (100, 200, 0), 3)
            cv2.putText(img, str(i), ((x + r), y), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 2)
        if settings.DEBUG:
            print("Spike contour #%d" % i)
            print("\tSize: %d" % size)


    return blobs_spikes


def load_blobs(img, img_blobs, img_spikes):
    """Lower level function"""

    # Analyze spikes
    contours_spikes = get_contours(img_spikes)
    blobs_spikes = find_spikes(contours_spikes, img)
    hide_spikes(img_blobs, blobs_spikes)

    if settings.DEBUG and len(blobs_spikes) > 0:
        show(img_blobs)

    # Analyze blobs
    contours = get_contours(img_blobs)
    check_concavity(img, contours)
    blobs, my_blob, blobs_bbox = find_blobs(contours, blobs_spikes, img)

    if settings.DEBUG:
        print("Found %d objects." % len(contours))

    cv2.rectangle(img, (blobs_bbox[0], blobs_bbox[1]), (blobs_bbox[2], blobs_bbox[3]), (0, 255, 0), 4)
    if settings.DEBUG:
        show(img)

    return blobs, my_blob, blobs_spikes, blobs_bbox


# ----------API--------------

def getBlobs():
    # Load image
    if settings.DEBUG:
        img = load_last_image(os.getcwd() + '\\..\\snapshots')
    else:
        img = quickGrab.screenGrab(settings.SAVE_IMAGES)

    # Pre-process image
    img_pre, img_spikes = preprocess(img)

    # Process image
    blobs, my_blob, spikes, bbox = load_blobs(img, img_pre, img_spikes)

    if my_blob and my_blob[2] < 50:
        if check_game_over(img):
            return None, None, None, None, img

    return blobs, my_blob, spikes, bbox, img


def main():
    t = time.time()
    getBlobs()
    print("\nRun time: %f seconds" % (time.time() - t))


if __name__ == '__main__':
    main()

