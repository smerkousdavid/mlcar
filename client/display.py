import tkFileDialog
import urllib
from Tkinter import *
from datetime import datetime
from threading import Thread
from time import sleep
from ttk import Style, Label, OptionMenu, Button

import cv2
import numpy as np
from PIL import Image, ImageTk
from nanomsg import NanoMsgAPIError, NanoMsgError

from mlcar.ai import AI, save_model_location
from mlcar.communicate import Communicate
from mlcar.configs import load_configs, get_configs, save_configs, linear_map
from mlcar.controller import Controller
from mlcar.data import DataLogger
from mlcar.logger import Logger

log = Logger("display")

coms = None
configs = None
video_frame = None
stream = None
stream_url = None
selector_var = None
selector_num = 0
selector_enable = True
min_h_scale = None
min_s_scale = None
min_v_scale = None
max_h_scale = None
max_s_scale = None
max_v_scale = None
blur_scale = None
min_angle_scale = None
max_angle_scale = None
min_h_var = None
min_s_var = None
min_v_var = None
max_h_var = None
max_s_var = None
max_v_var = None
min_angle_var = None
max_angle_var = None
generation_var = None
trial_var = None
enable_var = None
enabled_var = None
capture_var = None
prev_blur = 1
blur_var = None
width = None
height = None
frame = None
enabled = False
cont = None
agent = None
time_start = None
data_log = None


def clean_drive(current, future):
    """if current and future < 0.3:
        diff = abs(current - future)
        map = linear_map(diff, 0, 0.2, 0.15, 0.06)
        if map > 0.3:
            map = 0.3
        elif map < 0.06:
            map = 0.06
    else:
    """
    return linear_map(abs(future), 0, 0.9, 0.12, 0.095)


def clean_turn(from_camera):
    if 0.02 > from_camera > -0.02:
        return 0.0

    if from_camera > 0.4:
        return 0.4

    if from_camera < -0.4:
        return -0.4

    # sign = 1 if from_camera < 0 else -1
    # shift = -2 if from_camera < 0 else 2
    # return sign * (2 * (1 / (1 + abs(from_camera)))) + shift
    # return from_camera * (1 / (1 + abs(from_camera)))
    return from_camera * 0.8  # pow(from_camera, 3) + (0.5 * from_camera)


def static_drive(data):
    global cont
    power = clean_drive(data["cd"], data["fd"])
    turn = clean_turn(((-data["cd"] * 0.3) + (-data["fd"] * 1.6)) / 2.0)
    cont.drive(power, turn)
    return power, turn


def ai_drive(data):
    global cont, agent
    actions = agent.run(data["cd"], data["fd"])
    log.info("Predictions: %s" % str(actions))
    cont.drive(actions[0], actions[1])
    return actions[0], actions[1]


def communicate():
    global coms, cont, enabled, agent, capture_var, time_start, previous_future
    agent = AI()
    sleep(1)
    while True:
        try:
            data = coms.get()  # Get the data from the car
            if data is None:
                log.debug("Failed retrieving data from the car!")
            else:
                log.debug("Cpos: %.5f --- Fpos: %.5f" % (data["cd"], data["fd"]))
                if enabled:
                    power, turn = static_drive(data)
                    viewable = True
                    if data["cd"] == -2.0 or data["fd"] == -2.0 or \
                                    "0.09091" in ("%.5f" % data["cd"]) or \
                                    "0.09091" in ("%.5f" % data["fd"]):
                        log.error("AI went off course")
                        data["cd"] = -1.0
                        data["fd"] = -1.0
                        viewable = False
                        # agent.no_lane()
                    else:
                        pass
                        # ai_drive(data)
                    time_now = datetime.now()
                    t_disp = int((time_now - time_start).total_seconds() * 1000)
                    data_log.add_data([t_disp, int(viewable), data["cd"], abs(data["cd"]), data["fd"], abs(data["fd"]),
                                       power, abs(power), turn, abs(turn)])
                    capture_var.set(str(data_log.get_data_count()))
                else:
                    cont.drive(0.0, 0.0)
        except Exception as err:
            log.error(str(err))


def update_car(*args):
    global coms, configs, selector_num, min_h_var, min_s_var, min_v_var, max_h_var, max_s_var, max_v_var, blur_var, \
        prev_blur, min_angle_var, max_angle_var, enabled, enable_var, generation_var, trial_var
    min_h = min_h_var.get()
    min_s = min_s_var.get()
    min_v = min_v_var.get()
    max_h = max_h_var.get()
    max_s = max_s_var.get()
    max_v = max_v_var.get()
    blur = blur_var.get()
    min_angle = min_angle_var.get()
    max_angle = max_angle_var.get()
    enabled = bool(enable_var.get())

    try:
        generation = int(generation_var.get())
        trial = int(trial_var.get())
        configs["gens"]["num"] = generation
        configs["gens"]["trial"] = trial
    except:
        pass

    # Make the blur an odd number
    if blur % 2 == 0:
        if blur > prev_blur:
            blur += 1
        else:
            blur -= 1
        blur_var.set(str(blur))
        prev_blur = blur

    configs["vision"]["stream"] = selector_num
    configs["vision"]["min_h"] = min_h
    configs["vision"]["min_s"] = min_s
    configs["vision"]["min_v"] = min_v
    configs["vision"]["max_h"] = max_h
    configs["vision"]["max_s"] = max_s
    configs["vision"]["max_v"] = max_v
    configs["vision"]["blur"] = blur
    configs["vision"]["min_angle"] = min_angle
    configs["vision"]["max_angle"] = max_angle

    coms.send(selector_num, min_h, min_s, min_v, max_h, max_s, max_v, blur, min_angle, max_angle)

    save_configs(configs)

    sleep(0.01)


def update_select(*args):
    global selector_num, selector_enable, configs
    num = int(selector_var.get())
    selector_enable = (num != -1)
    configs["vision"]["stream"] = num
    log.info("Setting image source to %d" % num)
    selector_num = num
    update_car()


def exit_app():
    global stream, root
    try:
        stream.close()
    except:
        pass

    try:
        root.quit()
    except:
        pass
    log.info("Exiting...")
    exit(0)


def open_stream():
    global stream, stream_url
    while True:
        try:
            stream = urllib.urlopen(stream_url)
            break
        except Exception:
            log.error("Failed to connect! Retrying in 1 second...")
            sleep(1)


def pull_frame():
    global stream, stream_url, frame, mainframe, selector_enable
    fail_count = 0
    data = ''
    while True:
        try:
            if not selector_enable:
                sleep(0.5)
                continue
            data += stream.read(1024)
            a = data.find('\xff\xd8')
            b = data.find('\xff\xd9')
            if a != -1 and b != -1:
                jpg = data[a:b + 2]
                data = data[b + 2:]
                img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(img)

                if frame is None:
                    frame = Label(mainframe, image=img)
                    frame.image = img
                    frame.grid(column=1, row=0)
                else:
                    frame.configure(image=img)
                    frame.image = img

                if fail_count > 0:
                    fail_count -= 1
        except Exception as err:
            log.error("Failed pulling frame! (err: %s)" % str(err))
            if "Connection" in str(err) and "reset" in str(err):
                open_stream()
            fail_count += 10
            if fail_count > 900:
                open_stream()
                data = ''
                fail_count = 0
                # exit_app()


def save_model():
    global agent
    log.info("Saving model file...")
    try:
        agent.save()
    except Exception as err:
        log.error("Failed saving model %s" % str(err))


def load_model():
    global agent
    log.info("Showing load model selector...")
    while True:
        file_path = tkFileDialog.askopenfilename(initialdir=save_model_location,
                                                 title="Select ai model",
                                                 filetypes=(("all files", "*.*"),))
        if file_path is None:
            log.info("User did not select a valid model!")
            break

        try:
            agent.load(file_path)
            break
        except Exception as err:
            log.error("Failed loading model from! %s (err: %s)" % (file_path, str(err)))
            sleep(1)


def enable_car():
    global enable_var, enabled_var, generation_var, trial_var, data_log, time_start
    enable_var.set(True)
    enabled_var.set("Enabled")
    c_gen = int(generation_var.get())
    c_trial = int(trial_var.get())

    data_log.set_gen(c_gen)
    data_log.set_trial(c_trial)

    time_start = datetime.now()
    update_car()  # Send the new configurations to the car


def disable_car(complete):
    global enable_var, enabled_var, generation_var, trial_var, data_log, time_start, enabled
    if not bool(enable_var.get()):
        update_car()

    enabled = False
    enable_var.set(False)
    enabled_var.set("Disabled")

    # Final data sampling
    time_now = datetime.now()
    t_disp = int((time_now - time_start).total_seconds() * 1000)

    # Get the total count
    data_count = data_log.get_data_count()

    # Error sampling
    error_data = data_log.get_v_data("C")
    low_error, high_error = data_log.get_peaks(error_data)
    avg_error = data_log.get_average(error_data)
    median_error = data_log.get_median(error_data)
    a_low_error, a_high_error = data_log.get_peaks(error_data, True)
    a_avg_error = data_log.get_average(error_data, True)
    a_median_error = data_log.get_median(error_data, True)
    deviation_error = data_log.get_standard_deviation(error_data)

    # Power sampling
    power_data = data_log.get_v_data("G")
    low_power, high_power = data_log.get_peaks(power_data)
    avg_power = data_log.get_average(power_data)
    median_power = data_log.get_median(power_data)
    a_low_power, a_high_power = data_log.get_peaks(power_data, True)
    a_avg_power = data_log.get_average(power_data, True)
    a_median_power = data_log.get_median(power_data, True)
    deviation_power = data_log.get_standard_deviation(power_data)

    # Turn sampling
    turn_data = data_log.get_v_data("I")
    low_turn, high_turn = data_log.get_peaks(turn_data)
    avg_turn = data_log.get_average(turn_data)
    median_turn = data_log.get_median(turn_data)
    a_low_turn, a_high_turn = data_log.get_peaks(turn_data, True)
    a_avg_turn = data_log.get_average(turn_data, True)
    a_median_turn = data_log.get_median(turn_data, True)
    deviation_turn = data_log.get_standard_deviation(turn_data)

    f_data = data_log.get_data(2)
    f_data += [t_disp, data_count, avg_error, low_error, high_error, avg_power, low_power, high_power,
               avg_turn, low_turn, high_turn, int(complete)]
    data_log.set_data(2, f_data)

    # Advanced Error
    s_data = data_log.get_data(4)
    s_data += [None, "Absolute Average Error", "Absolute Median Error", "Absolute Low Error", "Absolute High Error",
               "Median Error"]
    data_log.set_data(4, s_data)

    g_data = data_log.get_data(5)
    g_data += [None, a_avg_error, a_median_error, a_low_error, a_high_error, median_error]
    data_log.set_data(5, g_data)

    # Advanced Power
    s_data = data_log.get_data(7)
    s_data += [None, "Absolute Average Power", "Absolute Median Power", "Absolute Low Power", "Absolute High Power",
               "Median Power"]
    data_log.set_data(7, s_data)

    g_data = data_log.get_data(8)
    g_data += [None, a_avg_power, a_median_power, a_low_power, a_high_power, median_power]
    data_log.set_data(8, g_data)

    # Advanced Turn
    s_data = data_log.get_data(10)
    s_data += [None, "Absolute Average Turn", "Absolute Median Turn", "Absolute Low Turn", "Absolute High Turn",
               "Median Turn"]
    data_log.set_data(10, s_data)

    g_data = data_log.get_data(11)
    g_data += [None, a_avg_turn, a_median_turn, a_low_turn, a_high_turn, median_turn]
    data_log.set_data(11, g_data)

    # Standard deviation
    s_data = data_log.get_data(13)
    s_data += [None, "Deviation Error", "Deviation Power", "Deviation Turn"]
    data_log.set_data(13, s_data)

    g_data = data_log.get_data(14)
    g_data += [None, deviation_error, deviation_power, deviation_turn]
    data_log.set_data(14, g_data)

    data_log.create_line_graph(title="Absolute", col_ref=["D", "F", "H", "J"], location="L16")
    data_log.create_line_graph(title="Scaled", col_ref=["B", "C", "E", "G", "I"], location="L46")

    # Save the spreadsheet data
    data_log.save_data()

    # Send the new configurations to the car
    update_car()


if __name__ == "__main__":
    log.info("Welcome to the mlcar camera stream display!\nLoading configs...")
    load_configs()
    configs = get_configs()
    stream_url = configs["display"]["camera_url"]
    width = configs["display"]["width"]
    height = configs["display"]["height"]
    data_log = DataLogger(configs["gens"]["data_file"])

    log.info("Attempting to connect to the controller")
    cont = Controller(configs)

    log.info("Attempting to connect to the car")
    coms = Communicate(configs)

    while True:
        try:
            coms.connect()
            break
        except (NanoMsgAPIError, NanoMsgError):
            log.error("Failed to connect! Retrying in 1 second...")
            sleep(1)

    log.info("Done!")
    log.info("Opening stream %s" % stream_url)
    open_stream()

    log.info("Done!")
    log.info("Starting mjpeg client!")
    Thread(target=pull_frame).start()

    root = Tk()
    root.title("mlcar control panel")
    Style(root).theme_use("clam")

    mainframe = Frame(root)
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)
    mainframe.pack(pady=10, padx=10)

    selector_var = StringVar(root)
    selector_var.set(0)
    selector = OptionMenu(mainframe, selector_var, "0", "-1", "0", "1", "2", "3")
    Label(mainframe, text="Image Source").grid(column=1, row=1)
    selector.grid(column=1, row=2)
    selector_num = configs["vision"]["stream"]
    selector_var.set(str(configs["vision"]["stream"]))
    selector_var.trace('w', update_select)

    min_h_var = IntVar(root)
    min_s_var = IntVar(root)
    min_v_var = IntVar(root)
    max_h_var = IntVar(root)
    max_s_var = IntVar(root)
    max_v_var = IntVar(root)
    blur_var = IntVar(root)
    min_angle_var = IntVar(root)
    max_angle_var = IntVar(root)
    enable_var = IntVar(root)
    generation_var = StringVar(root)
    trial_var = StringVar(root)
    enabled_var = StringVar(root)
    capture_var = StringVar(root)

    # MIN HSV
    Label(mainframe, text="Min H").grid(column=0, row=1)
    min_h_scale = Scale(mainframe, from_=0, to=255, showvalue=True, orient=HORIZONTAL, variable=min_h_var)
    min_h_scale.grid(column=0, row=2)

    Label(mainframe, text="Min S").grid(column=0, row=3)
    min_s_scale = Scale(mainframe, from_=0, to=255, showvalue=True, orient=HORIZONTAL, variable=min_s_var)
    min_s_scale.grid(column=0, row=4)

    Label(mainframe, text="Min V").grid(column=0, row=5)
    min_v_scale = Scale(mainframe, from_=0, to=255, showvalue=True, orient=HORIZONTAL, variable=min_v_var)
    min_v_scale.grid(column=0, row=6)

    Label(mainframe, text="Blur").grid(column=0, row=7)
    blur_scale = Scale(mainframe, from_=1, to=30, showvalue=True, orient=HORIZONTAL, variable=blur_var)
    blur_scale.grid(column=0, row=8)

    # MAX HSV
    Label(mainframe, text="Max H").grid(column=2, row=1)
    max_h_scale = Scale(mainframe, from_=0, to=255, showvalue=True, orient=HORIZONTAL, variable=max_h_var)
    max_h_scale.grid(column=2, row=2)

    Label(mainframe, text="Max S").grid(column=2, row=3)
    max_s_scale = Scale(mainframe, from_=0, to=255, showvalue=True, orient=HORIZONTAL, variable=max_s_var)
    max_s_scale.grid(column=2, row=4)

    Label(mainframe, text="Max V").grid(column=2, row=5)
    max_v_scale = Scale(mainframe, from_=0, to=255, showvalue=True, orient=HORIZONTAL, variable=max_v_var)
    max_v_scale.grid(column=2, row=6)

    enable_button = Button(mainframe, text="Enable", command=enable_car)
    enable_button.grid(column=1, row=5)

    disable_button = Button(mainframe, text="Disable (Failure)", command=lambda: disable_car(False))
    disable_button.grid(column=1, row=6)

    enabled_var.set("Disabled")
    enabled_label = Label(mainframe, textvariable=enabled_var)
    enabled_label.grid(column=2, row=7)

    capture_var.set("0")
    capture_label = Label(mainframe, textvariable=capture_var)
    capture_label.grid(column=2, row=8)

    disable_complete_button = Button(mainframe, text="Disable (Complete Course)", command=lambda: disable_car(True))
    disable_complete_button.grid(column=2, row=9)

    Label(mainframe, text="Data and Modeling").grid(column=1, row=7)

    save_model_button = Button(mainframe, text="Save Model", command=save_model)
    save_model_button.grid(column=1, row=8)

    load_model_button = Button(mainframe, text="Load Model", command=load_model)
    load_model_button.grid(column=1, row=9)

    Label(mainframe, text="Generation").grid(column=1, row=10)
    generation_entry = Entry(mainframe, textvariable=generation_var)
    generation_entry.grid(column=1, row=11)

    Label(mainframe, text="Trial").grid(column=1, row=12)
    trial_entry = Entry(mainframe, textvariable=trial_var)
    trial_entry.grid(column=1, row=13)

    t_down_button = Button(mainframe, text="Trail Down", command=lambda: trial_var.set(str(int(trial_var.get()) - 1)))
    t_down_button.grid(column=0, row=13)

    t_up_button = Button(mainframe, text="Trial Up", command=lambda: trial_var.set(str(int(trial_var.get()) + 1)))
    t_up_button.grid(column=2, row=13)

    # LINE FILTERING
    """
    Label(mainframe, text="Min angle").grid(column=1, row=8)
    min_angle_scale = Scale(mainframe, from_=0, to=90, showvalue=True, orient=HORIZONTAL, variable=min_angle_var)
    min_angle_scale.grid(column=1, row=9)

    Label(mainframe, text="Max angle").grid(column=1, row=10)
    max_angle_scale = Scale(mainframe, from_=0, to=90, showvalue=True, orient=HORIZONTAL, variable=max_angle_var)
    max_angle_scale.grid(column=1, row=11)
    """

    min_h_var.set(configs["vision"]["min_h"])
    min_s_var.set(configs["vision"]["min_s"])
    min_v_var.set(configs["vision"]["min_v"])
    max_h_var.set(configs["vision"]["max_h"])
    max_s_var.set(configs["vision"]["max_s"])
    max_v_var.set(configs["vision"]["max_v"])
    blur_var.set(configs["vision"]["blur"])
    min_angle_var.set(configs["vision"]["min_angle"])
    max_angle_var.set(configs["vision"]["max_angle"])
    generation_var.set(str(configs["gens"]["num"]))
    trial_var.set(str(configs["gens"]["trial"]))

    min_h_var.trace('w', update_car)
    min_s_var.trace('w', update_car)
    min_v_var.trace('w', update_car)
    max_h_var.trace('w', update_car)
    max_s_var.trace('w', update_car)
    max_v_var.trace('w', update_car)
    blur_var.trace('w', update_car)
    min_angle_var.trace('w', update_car)
    max_angle_var.trace('w', update_car)
    enable_var.trace('w', update_car)
    generation_var.trace('w', update_car)
    trial_var.trace('w', update_car)

    update_car()

    exit_button = Button(mainframe, text="Exit", command=exit_app)
    exit_button.grid(column=1, row=3)

    update_button = Button(mainframe, text="Push Update", command=update_car)
    update_button.grid(column=1, row=4)

    log.info("Done!")
    log.info("Starting the machine learning thread!")
    Thread(target=communicate).start()

    root.mainloop()
