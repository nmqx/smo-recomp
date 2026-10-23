"""A small window to choose the picture and window options, then play.

    python scripts/launcher.py            (or pythonw: no console)

It changes only the keys it shows in build/config/graphics.ini (the file the game reads at start, gfx_options.c; its
comments and every other key are kept) and starts the game as scripts/run.py play does (log in build/run/play.log;
F12 saves the frame's render passes into build/passshots). The picture size is the game's internal resolution:
resolution_scale times the game's 1600x900 (above 1.0 the GPU draws more pixels, the CPU does the same work).
"""
import ctypes
import os
import re
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tools'))
import target as T  # noqa: E402

INI = os.path.join(T.build('config'), 'graphics.ini')
GAME_H = 900                                       # the game's own picture is 1600x900

# (label, resolution_scale); 'screen' = the monitor's height / 900
PICTURES = [('720p (1280x720) - le plus léger', 0.8), ('900p (1600x900) - celle du jeu', 1.0), ('1080p (1920x1080)', 1.2),
            ('1440p (2560x1440)', 1.6), ('4K (3840x2160)', 2.4), ("Hauteur de l'écran", 'screen')]
WINDOWS = [('Plein écran sans bordure', 'borderless_fullscreen', None), ('Fenêtre 1280x720', 'borderless_window', (1280, 720)),
           ('Fenêtre 1600x900', 'borderless_window', (1600, 900)), ('Fenêtre 1920x1080', 'borderless_window', (1920, 1080)),
           ('Fenêtre 2560x1440', 'borderless_window', (2560, 1440))]


def read_ini():
    keys = {}
    if os.path.exists(INI):
        for line in open(INI, encoding='utf-8', errors='replace'):
            m = re.match(r'\s*([a-z_]+)\s*=\s*(.*?)\s*$', line)
            if m:
                keys[m.group(1)] = m.group(2)
    return keys


def write_ini(values):
    """Sets these keys in graphics.ini, keeping everything else; keys missing from the file are added at its end."""
    lines = open(INI, encoding='utf-8', errors='replace').read().split('\n') if os.path.exists(INI) else []
    left = dict(values)
    for i, line in enumerate(lines):
        m = re.match(r'(\s*)([a-z_]+)(\s*=\s*)', line)
        if m and m.group(2) in left:
            lines[i] = '%s%s%s%s' % (m.group(1), m.group(2), m.group(3), left.pop(m.group(2)))
    lines += ['%s = %s' % kv for kv in left.items()]
    os.makedirs(os.path.dirname(INI), exist_ok=True)
    with open(INI, 'w', encoding='utf-8', newline='') as f:
        f.write('\r\n'.join(l.rstrip('\r') for l in lines))


def screen_size():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # real pixels, not the scaled desktop's
    except Exception:
        pass
    u = ctypes.windll.user32
    return u.GetSystemMetrics(0), u.GetSystemMetrics(1)


def main():
    sw, sh = screen_size()
    keys = read_ini()
    root = tk.Tk()
    root.title('Super Mario Odyssey - options')
    root.resizable(False, False)
    f = ttk.Frame(root, padding=16)
    f.grid()

    def row(r, text, widget):
        ttk.Label(f, text=text).grid(row=r, column=0, sticky='w', pady=4, padx=(0, 12))
        widget.grid(row=r, column=1, sticky='we', pady=4)

    pic_labels = [p[0] for p in PICTURES]
    scale_now = float(keys.get('resolution_scale', '1.0') or 1.0)
    pic_now = next((p[0] for p in PICTURES if p[1] != 'screen' and abs(p[1] - scale_now) < 0.01), PICTURES[5][0])
    pic = tk.StringVar(value=pic_now)
    row(0, 'Résolution de rendu', ttk.Combobox(f, textvariable=pic, values=pic_labels, state='readonly', width=34))

    win_labels = [w[0] for w in WINDOWS]
    if keys.get('window') in ('borderless_fullscreen', 'fullscreen'):
        win_now = WINDOWS[0][0]
    else:
        size = (int(keys.get('width', 1280) or 1280), int(keys.get('height', 720) or 720))
        win_now = next((w[0] for w in WINDOWS if w[2] == size), WINDOWS[1][0])
    win = tk.StringVar(value=win_now)
    row(1, 'Affichage', ttk.Combobox(f, textvariable=win, values=win_labels, state='readonly', width=34))

    interp = tk.BooleanVar(value=keys.get('frame_interpolation', 'off') in ('optical_flow', 'on'))
    row(2, 'Interpolation 120 fps (NVIDIA)', ttk.Checkbutton(f, variable=interp))
    fps = tk.BooleanVar(value=keys.get('frame_time_overlay', 'off') in ('on', '1'))
    row(3, 'Compteur FPS', ttk.Checkbutton(f, variable=fps))
    aniso = tk.BooleanVar(value=keys.get('anisotropic_filtering', 'game') == '16')
    row(4, 'Filtrage anisotrope 16x', ttk.Checkbutton(f, variable=aniso))

    info = ttk.Label(f, text='', foreground='#555')
    info.grid(row=5, column=0, columnspan=2, sticky='w', pady=(8, 0))

    def scale_of(label):
        s = dict(PICTURES)[label]
        return round(sh / GAME_H, 3) if s == 'screen' else s

    def update(*_):
        s = scale_of(pic.get())
        info.configure(text='Rendu %dx%d, écran %dx%d. Le jeu est en 16:9 : bandes noires sur un écran plus large.'
                       % (round(1600 * s), round(900 * s), sw, sh))
    pic.trace_add('write', update)
    update()

    def save():
        w = dict((x[0], x) for x in WINDOWS)[win.get()]
        values = {'resolution_scale': ('%.3f' % scale_of(pic.get())).rstrip('0').rstrip('.'), 'window': w[1],
                  'frame_interpolation': 'optical_flow' if interp.get() else 'off',
                  'frame_time_overlay': 'on' if fps.get() else 'off',
                  'anisotropic_filtering': '16' if aniso.get() else 'game'}
        if w[2]:
            values['width'], values['height'] = str(w[2][0]), str(w[2][1])
        write_ini(values)

    def play():
        save()
        shots = os.path.join(T.build('passshots'))
        os.makedirs(shots, exist_ok=True)
        cmd = [sys.executable, os.path.join(REPO, 'scripts', 'run.py'), 'play', '--exe', 'smo_native.exe', '--seconds', '0',
               '--env', 'SMO_VK_PASSSHOT_KEY=1', '--env', 'SMO_SHOT_DIR=' + shots.replace('\\', '/')]
        subprocess.Popen(cmd, cwd=REPO, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        root.destroy()

    b = ttk.Frame(f)
    b.grid(row=6, column=0, columnspan=2, sticky='e', pady=(12, 0))
    ttk.Button(b, text='Enregistrer', command=lambda: (save(), root.destroy())).grid(row=0, column=0, padx=4)
    ttk.Button(b, text='Jouer', command=play).grid(row=0, column=1)
    root.mainloop()


if __name__ == '__main__':
    main()
