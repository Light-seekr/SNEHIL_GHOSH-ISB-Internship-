"""
finger_mic_volume_fixed.py
Windows-only. Fixed-size Tkinter window, no FPS, shows mute icon.

- Child process (COM/pycaw) controls microphone endpoint (eCapture).
- MediaPipe hand detection counts fingers (single hand).
- Smoothing: require stable_frames consecutive frames before applying.
- Mapping: 0->0%, 1->20%, 2->40%, 3->60%, 4->80%, 5->100%.
"""

import sys
import os
import tempfile
import subprocess
import threading
import queue
import time
import traceback
from collections import deque, Counter

# Platform guard
import platform
if platform.system() != "Windows":
    raise SystemExit("This script runs only on Windows (requires pycaw/comtypes).")

# --- Child code string (the child process handles COM/pycaw for mic endpoint) ---
child_code = r'''
# child_mic_server.py -- handles COM/pycaw in isolated process for microphone endpoint (eCapture)
import sys, traceback
def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

try:
    from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
    from comtypes.client import CreateObject
    from comtypes import GUID
    from ctypes import POINTER, cast
    from pycaw.pycaw import IAudioEndpointVolume, IMMDeviceEnumerator
except Exception as ex:
    print("ERR IMPORT", ex)
    eprint("IMPORT TRACEBACK:")
    traceback.print_exc()
    sys.exit(2)

def out(s):
    print(s)
    sys.stdout.flush()

try:
    CoInitialize()
except Exception as ex:
    out("ERR COINIT " + str(ex))
    eprint("COINIT TRACEBACK:")
    traceback.print_exc()
    sys.exit(3)

def _create_enum():
    try:
        from comtypes.client import CreateObject
        from pycaw.pycaw import IMMDeviceEnumerator
        try:
            return CreateObject("MMDeviceEnumerator.MMDeviceEnumerator", interface=IMMDeviceEnumerator)
        except Exception:
            clsid = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
            return CreateObject(clsid, interface=IMMDeviceEnumerator)
    except Exception:
        raise

def _get_vol_iface():
    enumerator = _create_enum()
    # eCapture = 1 (microphone), role eConsole = 0
    device = enumerator.GetDefaultAudioEndpoint(1, 0)
    iface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    from ctypes import cast, POINTER
    vol = cast(iface, POINTER(IAudioEndpointVolume))
    return vol

try:
    vol = _get_vol_iface()
except Exception as ex:
    out("ERR GET_IFACE " + str(ex))
    eprint("GET_IFACE TRACEBACK:")
    traceback.print_exc()
    try:
        CoUninitialize()
    except:
        pass
    sys.exit(4)

out("OK READY")

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    try:
        low = line.lower()
        if low == "get":
            try:
                cur = vol.GetMasterVolumeLevelScalar()
                p = int(round(cur * 100))
                out(f"OK GET {p}")
            except Exception as ex:
                out("ERR GET " + str(ex))
                eprint("GET TRACE:")
                traceback.print_exc()
        elif low.startswith("set:"):
            try:
                arg = line.split(":",1)[1].strip()
                v = int(arg)
                v = max(0, min(100, v))
                vol.SetMasterVolumeLevelScalar(v / 100.0, None)
                out(f"OK SET {v}")
            except Exception as ex:
                out("ERR SET " + str(ex))
                eprint("SET TRACE:")
                traceback.print_exc()
        elif low in ("mute","toggle_mute"):
            try:
                curmute = bool(vol.GetMute())
                if low == "mute":
                    vol.SetMute(1, None)
                    out("OK MUTE 1")
                else:
                    vol.SetMute(0 if curmute else 1, None)
                    out(f"OK TOGGLE {0 if curmute else 1}")
            except Exception as ex:
                out("ERR MUTE " + str(ex))
                eprint("MUTE TRACE:")
                traceback.print_exc()
        elif low == "ismute":
            try:
                curmute = int(bool(vol.GetMute()))
                out(f"OK ISMUTE {curmute}")
            except Exception as ex:
                out("ERR ISMUTE " + str(ex))
                eprint("ISMUTE TRACE:")
                traceback.print_exc()
        elif low == "quit":
            out("OK QUIT")
            break
        else:
            out("ERR UNKNOWN " + line)
    except Exception as ex:
        out("ERR LOOP " + str(ex))
        eprint("LOOP TRACE:")
        traceback.print_exc()
        break

try:
    CoUninitialize()
except Exception:
    pass
'''

# Write child to temp file
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py", prefix="child_mic_server_")
tmp.write(child_code.encode("utf-8"))
tmp.flush()
tmp.close()
child_path = tmp.name
print("Child script written to:", child_path)

# Launch child process unbuffered (-u)
proc = subprocess.Popen([sys.executable, "-u", child_path],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, bufsize=1)

stdout_q = queue.Queue()
stderr_q = queue.Queue()

def _read_stream(stream, q):
    try:
        for line in stream:
            q.put(line.rstrip("\n"))
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except:
            pass

t_o = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_q), daemon=True)
t_e = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_q), daemon=True)
t_o.start(); t_e.start()

# Wait for OK READY
start = time.time(); ready = False
while time.time() - start < 6.0:
    try:
        line = stdout_q.get(timeout=0.2)
    except queue.Empty:
        continue
    print("CHILD:", line)
    if line.strip().upper().startswith("OK READY"):
        ready = True
        break

if not ready:
    while not stderr_q.empty():
        print("CHILD STDERR:", stderr_q.get())
    try: proc.kill()
    except: pass
    raise SystemExit("Child did not start correctly; check stderr above.")

# MicController: talk to child
class MicController:
    def __init__(self, proc, stdout_q):
        self.proc = proc
        self.stdout_q = stdout_q
        self.lock = threading.Lock()

    def _send(self, line, timeout=2.0):
        with self.lock:
            if self.proc.poll() is not None:
                raise RuntimeError("Child process has exited.")
            try:
                self.proc.stdin.write(line.strip() + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                raise RuntimeError("Failed to write to child stdin: " + str(e))
            start = time.time()
            while time.time() - start < timeout:
                try:
                    out = self.stdout_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if out is None:
                    continue
                return out
            raise TimeoutError("No reply from child")

    def get_volume(self):
        out = self._send("get")
        if out.upper().startswith("OK GET"):
            try:
                return int(out.split()[-1])
            except:
                return 50
        raise RuntimeError("Child error: " + out)

    def set_volume(self, percent):
        out = self._send(f"set:{int(percent)}")
        return out

    def toggle_mute(self):
        out = self._send("toggle_mute")
        return out

    def mute(self):
        out = self._send("mute")
        return out

    def is_muted(self):
        out = self._send("ismute")
        if out.upper().startswith("OK ISMUTE"):
            try:
                return bool(int(out.split()[-1]))
            except:
                return False
        raise RuntimeError("Child error: " + out)

    def close(self):
        try:
            self._send("quit", timeout=1.0)
        except Exception:
            pass
        try:
            self.proc.kill()
        except:
            pass
        try:
            self.proc.wait(timeout=1.0)
        except:
            pass

mc = MicController(proc, stdout_q)

def get_mic_volume_percent():
    try:
        return mc.get_volume()
    except Exception as e:
        print("get_mic_volume_percent error:", e)
        return 50

def set_mic_volume_percent(v):
    try:
        mc.set_volume(int(v))
    except Exception as e:
        print("set_mic_volume_percent error:", e)

def get_mic_is_muted():
    try:
        return mc.is_muted()
    except Exception as e:
        print("get_mic_is_muted error:", e)
        return False

# --- Main program: MediaPipe + OpenCV + Tkinter GUI ---
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import tkinter as tk

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers_from_landmarks(hand_landmarks, handedness_label=None):
    if hand_landmarks is None:
        return 0
    lm = hand_landmarks.landmark
    fingers = 0
    tips_pips = [(8,6), (12,10), (16,14), (20,18)]
    for tip, pip in tips_pips:
        try:
            if lm[tip].y < lm[pip].y:
                fingers += 1
        except Exception:
            pass
    # thumb
    try:
        thumb_tip_x = lm[4].x
        thumb_mcp_x = lm[2].x
        label = None
        if handedness_label and hasattr(handedness_label, 'classification') and len(handedness_label.classification) > 0:
            label = handedness_label.classification[0].label
        if label == "Right":
            if thumb_tip_x < thumb_mcp_x - 0.03:
                fingers += 1
        else:
            if thumb_tip_x > thumb_mcp_x + 0.03:
                fingers += 1
    except Exception:
        pass
    return max(0, min(5, fingers))

def fingers_to_percent(n):
    n = max(0, min(5, int(n)))
    if n == 0:
        return 0
    return n * 20

class App:
    def __init__(self, stable_frames=6, cam_index=0, win_w=960, win_h=640):
        self.stable_frames = stable_frames
        self.cam_index = cam_index
        # Open camera (prefer DirectShow)
        try:
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        except Exception:
            self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera. Check camera index and permissions.")
        # MediaPipe hands
        self.hands = mp_hands.Hands(static_image_mode=False, model_complexity=1,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                    max_num_hands=1)
        self.history = deque(maxlen=self.stable_frames)
        self.current_applied = get_mic_volume_percent()
        self.last_observed = 0
        self.running = True

        # Tkinter fixed-size window
        self.root = tk.Tk()
        self.root.title("Finger -> Mic Volume")
        self.win_w = win_w
        self.win_h = win_h
        self.root.geometry(f"{self.win_w}x{self.win_h}")
        self.root.resizable(False, False)  # FIXED SIZE
        self.root.configure(bg="#111111")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Video label
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.place(x=0, y=0, width=self.win_w, height=self.win_h)

        # Start update loop
        self.root.after(10, self._update_frame)

    def _on_close(self):
        self.running = False
        try:
            self.cap.release()
        except:
            pass
        try:
            self.hands.close()
        except:
            pass
        try:
            mc.close()
        except:
            pass
        try:
            os.unlink(child_path)
        except:
            pass
        try:
            self.root.destroy()
        except:
            pass
        sys.exit(0)

    def _overlay_text(self, frame, fingers, volume, muted):
        h, w = frame.shape[:2]
        txt1 = f"Fingers: {fingers}"
        txt2 = f"Mic: {volume}%"
        mute_icon = "🔇" if muted else "🔊"
        txt3 = f"{mute_icon}"
        # Text params
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale1 = max(0.9, w / 640)
        scale2 = max(0.6, w / 900)
        thickness1 = 3
        thickness2 = 2
        # compute sizes
        (t1_w, t1_h), _ = cv2.getTextSize(txt1, font, scale1, thickness1)
        (t2_w, t2_h), _ = cv2.getTextSize(txt2, font, scale2, thickness2)
        pad = 12
        box_w = max(t1_w, t2_w) + pad*4
        box_h = t1_h + t2_h + pad*3
        # draw overlay rectangle with alpha
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (6,6,6), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # draw texts
        org1 = (10 + pad, 10 + pad + t1_h)
        cv2.putText(frame, txt1, org1, font, scale1, (0,220,180), thickness1+2, cv2.LINE_AA)
        cv2.putText(frame, txt1, org1, font, scale1, (255,255,255), thickness1, cv2.LINE_AA)
        org2 = (10 + pad, 10 + pad + t1_h + pad + t2_h)
        cv2.putText(frame, txt2, org2, font, scale2, (0,180,220), thickness2+2, cv2.LINE_AA)
        cv2.putText(frame, txt2, org2, font, scale2, (255,255,255), thickness2, cv2.LINE_AA)
        # mute icon at top-right
        icon_scale = max(1.0, w / 640)
        icon_thick = 2
        (i_w, i_h), _ = cv2.getTextSize(txt3, font, icon_scale, icon_thick)
        icon_org = (w - i_w - 16, 16 + i_h)
        cv2.putText(frame, txt3, icon_org, font, icon_scale, (255,255,255), icon_thick+2, cv2.LINE_AA)
        cv2.putText(frame, txt3, icon_org, font, icon_scale, (0,0,0) if muted else (0,200,0), icon_thick, cv2.LINE_AA)
        return frame

    def _update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._update_frame)
            return
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = None
        try:
            results = self.hands.process(img_rgb)
        except Exception:
            results = None

        fingers_found = 0
        handedness_label = None
        if results and getattr(results, "multi_hand_landmarks", None):
            if len(results.multi_hand_landmarks) > 0:
                hand_landmarks = results.multi_hand_landmarks[0]
                try:
                    if getattr(results, "multi_handedness", None) and len(results.multi_handedness) > 0:
                        handedness_label = results.multi_handedness[0]
                except Exception:
                    handedness_label = None
                fingers_found = count_fingers_from_landmarks(hand_landmarks, handedness_label)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            fingers_found = 0

        # smoothing
        self.history.append(fingers_found)
        try:
            if len(self.history) == self.history.maxlen:
                counts = Counter(self.history)
                most_common = counts.most_common()
                if most_common:
                    max_count = most_common[0][1]
                    candidates = [v for v,c in most_common if c == max_count]
                    chosen = max(candidates)
                else:
                    chosen = self.history[-1]
            else:
                chosen = self.history[-1]
        except Exception:
            chosen = self.history[-1] if self.history else 0

        target_pct = fingers_to_percent(chosen)
        if len(self.history) == self.history.maxlen and target_pct != self.current_applied:
            try:
                set_mic_volume_percent(target_pct)
                time.sleep(0.02)
                self.current_applied = get_mic_volume_percent()
                # small console log
                print(f"[APPLY] Fingers {chosen} -> {target_pct}%")
            except Exception as e:
                print("Error applying mic percent:", e)

        self.last_observed = chosen
        muted = get_mic_is_muted()
        display_frame = self._overlay_text(frame.copy(), self.last_observed, self.current_applied, muted)

        # convert to PIL then ImageTk and display
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        try:
            pil = pil.resize((self.win_w, self.win_h), Image.Resampling.LANCZOS)
        except Exception:
            pil = pil.resize((self.win_w, self.win_h), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=pil)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(30, self._update_frame)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            try:
                mc.close()
            except:
                pass

if __name__ == "__main__":
    try:
        app = App(stable_frames=6, cam_index=0, win_w=960, win_h=640)
        app.run()
    except Exception as e:
        print("Fatal error in main:", e)
        traceback.print_exc()
        try:
            mc.close()
        except:
            pass
        try:
            os.unlink(child_path)
        except:
            pass
        sys.exit(1)
