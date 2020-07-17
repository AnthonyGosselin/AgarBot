import time
from win32 import win32api
from win32.lib import win32con
import math
from src import agarBlobs, quickGrab, settings
from cv2 import arrowedLine
import os
import shutil
import datetime
import keyboard

# TODO : Optimize feeding technique
# TODO : Better player identification (differentiating blobs who are in contact)
#   todo : realize when we can't find player
# TODO : Detect pop-ups at beggining and end of game and dismiss them


curr_log = None


class Log:
    data = []

    def log(self, *msg):
        full_msg = ""
        for text in msg:
            full_msg = full_msg + ' ' + str(text)
        self.data.append(full_msg)
        if settings.VERBOSE:
            print(full_msg)

    def save(self, location):
        f = open(location + '\\log.txt', "w")
        for msg in self.data:
            f.write(msg + "\n")
        f.close()


def new_log(*msg):
    if curr_log:
        new_log(msg)

def mouse_pos(pos):
    win32api.SetCursorPos(pos)


def left_click(pos):
    mouse_pos(pos)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)


def click_play_button():
    left_click(settings.PLAY_BUTTON)


def click_continue_button():
    left_click(settings.CONTINUE_BUTTON)


def blob_dist(b1, b2):
    return math.sqrt((b1[0] - b2[0]) ** 2 + (b1[1] - b2[1]) ** 2)


def rel_dist(b):
    return math.sqrt((b[0]) ** 2 + (b[1]) ** 2)


def copy_dir(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copy_dir(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def save_game():
    if settings.SAVE_GAME:
        game_path = settings.SAVED_GAMES_PATH + '\\z_Game_' + str(int(time.time()))
        os.mkdir(game_path)
        copy_dir(settings.SCREEN_GRABS_PATH, game_path)
        curr_log.save(game_path)


def bbox_dists(bbox, blob):
    x, y, size, r = blob
    dists = [
            x - bbox[0],
            y - bbox[1],
            bbox[2] - x,
            bbox[3] - y
            ]

    sum_vect = [0, 0]

    for i in range(4):
        if dists[i] < settings.OUTER_RANGE:
            closeness_fact = (settings.OUTER_RANGE / dists[i]) * settings.OUTER_FACTOR
            if i == 0:
                sum_vect[0] += (1) * closeness_fact
            elif i == 1:
                sum_vect[1] += (1) * closeness_fact
            elif i == 2:
                sum_vect[0] += (-1) * closeness_fact
            elif i == 3:
                sum_vect[1] += (-1) * closeness_fact

    if sum_vect != [0, 0]:
        new_log("bbox vect: ", sum_vect)

    return sum_vect, min(dists)


def rel_blob(blob, my_blob):
    (my_x, my_y, my_size, my_r) = my_blob
    (x, y, size, r) = blob
    rel_size = float(size / my_size)
    rel_x = x - my_x
    rel_y = y - my_y
    return (rel_x, rel_y, rel_size, r)


def split_blobs(blobs, spikes, my_blob):

    smaller = []
    bigger = []
    rel_spikes = []
    for blob in blobs:
        rel = rel_blob(blob, my_blob)
        if rel[2] >= 1.00:
            bigger.append(rel)
        else:
            smaller.append(rel)

    for blob in spikes:
        rel = rel_blob(blob, my_blob)
        # Only keep if smaller
        if rel[2] < 1.00:
            rel_spikes.append(rel)

    return smaller, bigger, rel_spikes


def get_vector(blob, range, fact=1):
    # Normalize vector and apply closeness factor
    (x, y, size, r) = blob
    dist = abs(rel_dist(blob) - r)  # Must take radius of enemy in account

    # Normalize vector and apply closeness factor
    closeness_fact = (range / dist) * fact
    magn_vect = math.sqrt(x ** 2 + y ** 2)
    vect_x = (-x / magn_vect) * closeness_fact
    vect_y = (-y / magn_vect) * closeness_fact

    return [vect_x, vect_y]


def move_dir(smaller, bigger, smaller_spikes, player_blob, blobs_bbox):
    sum_vect = [0, 0]
    closest_dist = 9999
    # Add vectors for enemies
    for blob in bigger:
        vect = get_vector(blob, settings.DANGER_RANGE, settings.BLOB_FACTOR)
        sum_vect[0] += vect[0]
        sum_vect[1] += vect[1]
        new_log("\tenemy vect: ", vect)

        # Check for closest
        (x, y, size, r) = blob
        dist = abs(rel_dist(blob) - r)  # Must take radius of enemy in account
        if dist < closest_dist:
            closest_dist = dist

    # Add vectors for out of bounds
    bbox_vect, bbox_closest = bbox_dists(blobs_bbox, player_blob)
    sum_vect[0] += bbox_vect[0]
    sum_vect[1] += bbox_vect[1]

    # Add vectors for smaller spikes
    closest_spike = 9999
    for spike in smaller_spikes:
        vect = get_vector(spike, settings.SPIKES_RANGE, settings.SPIKES_FACTOR)
        sum_vect[0] += vect[0]
        sum_vect[1] += vect[1]
        new_log("\tspike vect: ", vect)

        # Check for closest
        (x, y, size, r) = spike
        dist = abs(rel_dist(spike) - r)
        if dist < closest_spike:
            closest_spike = dist

    if closest_dist <= settings.DANGER_RANGE or closest_spike <= settings.SPIKES_RANGE or bbox_closest <= settings.OUTER_RANGE:
        # Flee mode
        new_log("Flee mode")
        # Normalize vector
        if sum_vect != [0, 0]:
            magn_vect = math.sqrt(sum_vect[0]**2 + sum_vect[1]**2)
            sum_vect[0] = (sum_vect[0]/magn_vect) * settings.MOUSE_DIST
            sum_vect[1] = (sum_vect[1]/magn_vect) * settings.MOUSE_DIST

    else:
        # Feed mode
        new_log("Feed mode")
        closest_dist = 9999
        closest_blob = -1
        ind = 0
        for blob in smaller:
            dist = rel_dist(blob)

            if dist < closest_dist and closest_blob < 99999:
                closest_dist = dist
                closest_blob = ind
            ind += 1
        if closest_blob != -1:
            # Normalize vector
            targ_blob = smaller[closest_blob]
            magn_vect = math.sqrt(targ_blob[0] ** 2 + targ_blob[1] ** 2)
            sum_vect[0] = (targ_blob[0]/magn_vect) * settings.MOUSE_DIST
            sum_vect[1] = (targ_blob[1]/magn_vect) * settings.MOUSE_DIST
        else:
            new_log("Nothing detected, waiting 1 second")
            time.sleep(1)

    new_log("Action:", sum_vect)

    return int(sum_vect[0] + settings.SCREEN_CENTER[0]), int(sum_vect[1] + settings.SCREEN_CENTER[1])


def compute_action():
    other_blobs, player_blob, spike_blobs, blobs_bbox, img = agarBlobs.getBlobs()

    # Check game over
    if not other_blobs and not player_blob:
        new_log("____________\nGAME OVER")
        return False, False
    if keyboard.is_pressed('c'):
        new_log("____________\nGAME CANCELLED")
        return False, True

    new_log("Player size:", player_blob[2])
    smaller_blobs, bigger_blobs, smaller_spikes = split_blobs(other_blobs, spike_blobs, player_blob)
    move_pos = move_dir(smaller_blobs, bigger_blobs, smaller_spikes, player_blob, blobs_bbox)
    mouse_pos(move_pos)

    if settings.DEBUG and img.any():
        arrowedLine(img, (player_blob[0], player_blob[1]), (move_pos[0], move_pos[1]), (255, 0, 0), 3)
        agarBlobs.show(img)
    if settings.SAVE_PROCESSED and img.any():
        if not settings.DEBUG:
            arrowedLine(img, (player_blob[0], player_blob[1]), (move_pos[0], move_pos[1]), (255, 0, 0), 3)
        quickGrab.save_cv(img)

    return True, False


def run_game():
    global curr_log
    curr_log = Log()

    new_log("Starting game (" + str(datetime.datetime.now()) + ")\n\n")
    quickGrab.empty_folder()
    click_play_button()
    time.sleep(1.5)

    run_times = []
    game_running = True
    while game_running:
        t = time.time()
        new_log("(%d) Run start:" % len(run_times), t)

        game_running, cancelled = compute_action()

        run_time = time.time() - t
        new_log("Run time: %f seconds\n\n" % run_time)
        run_times.append(run_time)

    # Stats
    new_log("Run time:")
    new_log("\tAvg:", sum(run_times)/len(run_times))
    new_log("\tMax:", max(run_times), "(ind: %d)" % run_times.index(max(run_times)))
    new_log("\tMin:", min(run_times), "(ind: %d)" % run_times.index(min(run_times)))
    new_log("\tTotal:", sum(run_times))

    if settings.SAVE_PROCESSED or settings.SAVE_IMAGES:
        save_game()

    return cancelled


def loop_run_game():
    cancelled = False
    while not cancelled:
        cancelled = run_game()
        click_continue_button()


if __name__ == '__main__':
    if settings.LOOP_PLAY:
        loop_run_game()
    else:
        run_game()


