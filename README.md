# smo-recomp

A native Windows x64 version of *Super Mario Odyssey* (1.4.1, or the 1.3.0 update), made by **static recompilation** of
the Nintendo Switch game, with its own runtime and a Vulkan renderer.

> **This fork adds:** the 1.3.0 update as a second accepted version (its program is the same NintendoSDK 7.3.2 build:
> 54,214 functions, every one lifted); **frame interpolation to 120 fps** with the GPU's optical flow engine (NVIDIA RTX
> 30 and later, `frame_interpolation = optical_flow`), with late frames of the game bridged by predicted frames; the
> runtime's own threads kept off the processors of the game's console cores; and a fix for the shader cache, which was
> never written on a fresh build. See [What this fork changes](#what-this-fork-changes) and
> [Building, step by step](#building-step-by-step).

It is **not an emulator**: there is no CPU interpreter and no JIT, and no console firmware runs. The game's AArch64
machine code is translated to C ahead of time and compiled for the PC; the system libraries, graphics API, audio,
video, input and file system the game uses are re-implemented by this project's runtime.

> **This repository contains no game code, no game assets, no keys and no firmware.**
> You need your own copy of the game, dumped from your own console. The build reads the dump's decrypted files: no
> keys and no firmware are needed. Everything derived from the game is generated on your PC, under `build/`, and never
> leaves it.

## Status

Early, not finished. Everything below was measured on the development PC (Windows 11, NVIDIA RTX 4090); where
something was not checked, it says so.

| Area | State |
|---|---|
| Start | boot, title screen, file select, Resume: play begins (a scripted start reaches the game's play state at frame 3,564 with the warm shader cache) |
| Cap Kingdom | `tests/run_cap_kingdom.py` plays from an empty save to Mario's first capture in Top-Hat Tower and checks 13 checkpoints against the game's own state: 13 of 13 (42,558 frames), 0 host exceptions, 0 Vulkan call failures, 0 operations not implemented |
| Every stage and scenario | `tests/stress_stages.py` started every stage the game plays as a stage scene (176) and every scenario of every kingdom (100), each from a save made to start there, played for a few seconds at 4x speed: 276 runs, 2.1 million frames, 0 Vulkan call failures, 0 operations not implemented, 0 freezes, 0 audio drops. Cutscenes, the Labo VR stages and the world map have scenes of their own and were not started this way; Luigi's Balloon World and the Koopa races are started by their characters and cannot be started from a save |
| Played by hand | from Cap Kingdom to Cascade Kingdom's first checkpoint, with an Xbox One controller¹; that save continues on 1.4.1 |
| Shaders | every stage in the game's shader files translates and compiles (Maxwell machine code → GLSL → SPIR-V), checked offline by `tests/check_all_shaders.py`: 27,888 of 27,888 distinct stages (7,381 vertex, 20,436 fragment, 3 geometry, 68 compute). The translator's structured control flow equals its reference form on the GPU: 1,519 of 1,519 stages over 6.2 million invocations¹ |
| Picture | the texture layout equals the one Nintendo's tools stored for all 20,203 textures of the RomFS (`tests/bntx_layout_check.py`). 0 frames with a single-frame flash in captures of every frame while the camera pans in Cascade Kingdom¹. The Ruined Kingdom's lightning (a geometry stage) draws: each bolt brightens about 3,000 pixels before the game draws its clouds over it¹ |
| Graphics memory | 1.4 GB at the end of the Cap Kingdom run, 2.0 GB in the Cookatiel rematch: textures the game gives up are freed once nothing names them |
| Audio | the game's own audio renderer runs (lifted from its system library) and plays through the Windows audio device (6 → 2 channel downmix, 48 kHz): 0 dropped samples, 0 underruns in the Cap Kingdom run |
| Video | H.264 through Windows Media Foundation: the tutorial videos play with their picture |
| Help | the pause menu's Action Guide opens inside the game window, scaled to it, from the manual in your dump (Microsoft Edge WebView2), and the game waits while it is open, as on the console; checked with scripted input in a 2560 × 1440 window: open, the Controls page with its pictures, back, Bonus, Choose Mode (the game's Options, then the page again), close. Without WebView2 it opens in your web browser |
| amiibo | F1, or both stick buttons on a pad, opens a menu drawn like the game's pause menu from its own layout textures, fonts and texts: 102 amiibo, in the game's language. The one chosen is held to a virtual NFC reader; checked in Cascade Kingdom: the game reads Peach and runs her help. Mario, Bowser and Luigi checked as well¹ |
| System calls | every system call the game made in the tested scenes is implemented (331 of 331 in the Cap Kingdom run); `SMO_IMPORT_SELFTEST=1` checks the ones written from the game's import list rather than met in play: 0 failures |
| Controllers | keyboard, XInput, and any pad through SDL3 (hot-plug, 8 players, remapping file, rumble, motion): 51 of 51 automated checks with virtual devices; played with an Xbox One controller; other physical pads not tried |
| Window and capture | a borderless window by default; F11 or a double-click switches to borderless fullscreen and back. The game keeps running at 60 frames a second while its window is not the active one or is minimised (`unfocused = pause` in `graphics.ini` stops it, as the console does). F12 or a pad's share button saves the frame to the save folder's `album` |
| Save data | journalled like the console's (two-phase commit); the game's autosave and Resume work |
| Speed | 60 fps, paced to a 60 Hz clock: at Cascade Kingdom's first checkpoint 59.8-60.0 fps in each of 20 windows of 300 frames (1% lows 49.4-56.0); the Cap Kingdom run averages 59.4 fps once play begins, lowest 49.5 while the tower stage loads. Development PC only |
| Online | not available: the network services answer as a console without a network |
| Not tried yet | what the 1.4.1 update adds: Balloon World, the new Snapshot Mode filters, the Labo VR mode |

¹ measured on the base game (1.0.0), before this project moved to 1.4.1; not measured again since.

## How it works

1. **Import.** `scripts/setup.py` reads your decrypted dump in place and converts the program's modules to ELF.
2. **Function list.** Every function of the executable, with boundaries from its own unwind tables and symbols (54,214
   in `main`).
3. **Lifting.** `tools/lifter/recomp.py` turns each AArch64 function into a C function; clang compiles them into a game
   DLL. A differential test runs sampled instructions both lifted and in the Unicorn emulator: 23,194 of 23,194 equal.
4. **Runtime.** `runtime/` stands in for the console: threads with the console's three-core scheduling, files and
   journalled saves, NVN graphics on Vulkan with a Maxwell shader translator, the audio renderer, controllers, movies,
   Help and the offline system services.

[`docs/architecture.md`](docs/architecture.md) explains each part.

## Requirements

- Windows 10/11 x64 and a Vulkan 1.3 GPU (developed on NVIDIA; `VK_EXT_graphics_pipeline_library`,
  `VK_EXT_extended_dynamic_state3`, `VK_EXT_vertex_input_dynamic_state` and `VK_EXT_depth_clip_control` are used).
- Your own *Super Mario Odyssey* dump at version 1.4.1 or 1.3.0: the program with the update, the RomFS with the update applied
  and the update's control data. [`docs/building.md`](docs/building.md) lists what to dump and how to lay it out.
- [LLVM/clang 9.0.0](https://releases.llvm.org/), Visual Studio 2019 Build Tools (16.11) with Windows SDK 10.0.19041,
  the [Vulkan SDK](https://vulkan.lunarg.com/) 1.4, [SDL3](https://github.com/libsdl-org/SDL) 3.4.18, the
  [WebView2 SDK](https://www.nuget.org/packages/Microsoft.Web.WebView2), Python 3.13.

## Quick start

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install lz4 capstone unicorn numpy Pillow
.venv\Scripts\python scripts\setup.py --dump "<folder of your dump>"
.venv\Scripts\python scripts\run.py play --exe smo_native.exe --seconds 0
```

The first build takes a while (lifting and compiling the game); later runs of `setup.py` only redo what changed.
[`docs/building.md`](docs/building.md) has the details, settings and troubleshooting.

## Building, step by step

Measured on Windows 11 with an RTX 3090 and a Ryzen 5 3600XT: about 6 minutes for a first build (lift 10 s, compile
about 3 minutes on 12 threads, link 1 minute).

1. **Dump your own game** from your own console with a homebrew dumper that writes decrypted files (nxdumptool), and lay
   the dump out in one folder:
   ```
   <dump>\
     exefs\          the program's ExeFS with the update (1.4.1 or 1.3.0): main, main.npdm, sdk, rtld, subsdk0
     exefs-1.0.0\    not needed any more (an older layout's base game ExeFS): ignored if present
     romfs\          the RomFS with the update applied (about 5.7 GB)
     control.nacp    the update's control data (its display version is checked: 1.4.1 or 1.3.0)
     manual\         optional: the Help pages (html-document\...)
     fonts\          optional: the console's shared fonts (.bfttf)
   ```
   Keep it on a fast drive: the build reads the RomFS in place.
2. **Install the tools** (paths are the defaults; each has a variable to point elsewhere):
   - Python 3.13 (`py -3.13`).
   - LLVM/clang **9.0.0** exactly: `LLVM-9.0.0-win64.exe` from [releases.llvm.org/9.0.0](https://releases.llvm.org/9.0.0/)
     (it is an NSIS archive: `7z x LLVM-9.0.0-win64.exe -oC:\llvm-9.0.0` unpacks it without installing), or set `LLVM_ROOT`.
   - Visual Studio Build Tools with the C++ workload and a Windows 10/11 SDK (2019 is the documented one; 2026 Build Tools
     with SDK 10.0.26100 also builds and links with 0 failures).
   - The [Vulkan SDK](https://vulkan.lunarg.com/) 1.4. If it is not under `C:\VulkanSDK`, set `VULKAN_HEADERS` to its
     `Include` folder.
   - SDL3 3.4.18: unzip `SDL3-devel-3.4.18-VC.zip` (from the SDL releases on GitHub) into `deps\SDL3-3.4.18`.
   - The WebView2 SDK: download `https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/1.0.2903.40` (a `.nupkg` is a
     zip) and unzip it into `deps\webview2-1.0.2903.40`.
3. **Build:**
   ```bat
   py -3.13 -m venv .venv
   .venv\Scripts\python -m pip install lz4 capstone unicorn numpy Pillow
   .venv\Scripts\python scripts\setup.py --only deps
   .venv\Scripts\python scripts\setup.py --dump "<dump>"
   ```
   `--only deps` names anything missing. The setup resumes where it stopped if interrupted; after changing only the
   runtime, `--only link` is enough. Optional, for a build tuned to your CPU (it then runs on that CPU family and later
   ones only): `set RECOMP_ARCH=-march=znver2 -mno-prfchw` (Zen 2; `-march=native` for whatever CPU builds it) before
   `setup.py`.
4. **Play:**
   ```bat
   .venv\Scripts\python scripts\run.py play --exe smo_native.exe --seconds 0
   ```
   Saves are in `build\userdata\save` (`Common.bin`, `File1.bin`). To use an existing save, put its files there. A test
   run's `--userdata <folder>` is a user-data root: its save goes in `<folder>\save\`, not in `<folder>` itself.
   Settings are in `build\config\graphics.ini` and `build\config\controllers.ini` (written on the first run). F11
   switches to fullscreen.
5. The first visit of each place compiles its shaders (loading screens wait for them); they are kept in `build\cache`
   and later visits load them.

## What this fork changes

- **1.3.0 accepted** (`tools/target.py` `VERSIONS`): the import, the generated tables and the setup report the dump's own
  version. 1.3.0's `main` is NintendoSDK 7.3.2 like 1.4.1's, and every rule of `tools/lifter/gen_tables.py` that finds
  the game functions the runtime needs gives exactly one function in it (the same addresses the base executable's
  symbols gave before), so the base game's ExeFS is not needed for 1.3.0 either.
- **Frame interpolation** (`runtime/gfx/nvn_vk_interp.c`, `frame_interpolation = optical_flow` in `graphics.ini`;
  `SMO_VK_INTERP=1/0` wins). Each frame of the game is copied out; the GPU's optical flow engine (`VK_NV_optical_flow`,
  on its own queue) measures the motion to the frame before in both directions on a 4x4 grid; a compute stage makes the
  frame between them (for every pixel the pair of samples joined best by the forward flow, the backward flow or no
  motion, so the 2D interface stays still); a presenter thread with a queue of its own shows the made frame, then the
  game's own half a frame later: 120 frames a second while the game runs at 60, about 8 ms later than without it.
  When the game's next frame is late, the presenter shows the newest frame moved on along its own motion instead
  (up to 6 times, `SMO_VK_INTERP_EXT_MAX`; `SMO_VK_INTERP_EXT=0` turns it off). Debug: `SMO_VK_INTERP_STATS=1`,
  `SMO_VK_INTERP_SHOT=<play frame>[,...]` (pictures of a made frame and its neighbours).
- **The runtime's own threads off the game's processors** (`runtime/os/os_threads.c` `os_host_cpu_mask`): the render
  thread, the pipeline workers and the presenter run on the processors the console-core model leaves free, not on those
  of the game's three console cores or their second hardware threads (`SMO_HOST_PIN=0` turns it off,
  `SMO_HOST_CPUS=<hex>` chooses).
- **Shader cache fix** (`runtime/gfx/nvn_vk_pipelines.c`): `..\cache\vk` was made with one `CreateDirectory` call,
  which fails when `build\cache` does not exist yet, so a fresh build compiled every shader again at every start.
- **Speed in heavy places** (measured in New Donk City's streets, about 2,000 draws a frame, Ryzen 5 3600XT, RTX 3090):

  | Change | fps there |
  |---|---|
  | as forked (lifted code and runtime at `-O1`) | about 20 |
  | lifted code at `-O2` (now the default, `RECOMP_OPT`), `RECOMP_ARCH=-march=znver2` | 30-36 |
  | registerSampler_'s free-slot search answered in host code (`smo_fast.c`; the game probes thousands of taken sampler slots per new sampler, each a lifted call: half the main thread's time there) | 38-42 |
  | present without waiting for the replay (`SMO_PRESENT_ASYNC`, now on by default; `=0` waits as before): the game draws the next frame while this one is replayed | 44-54 |

  The runtime is `-O2` too (`RECOMP_HOST_OPT`). `RECOMP_ARCH` (for example `-march=znver2`; add `-mno-prfchw` with
  clang 9 and a recent Windows SDK, whose `winnt.h` clashes with clang 9's `_m_prefetchw`) makes a build for that CPU
  family and later ones only. Changing these flags recompiles nothing by itself: run
  `tools\lifter\build_main.py <jobs> --force` and delete `build\obj\host` before `setup.py --only link`.
- **Resolution in heavy places**: the game's counter timestamps are taken when the render thread replays them, so the
  game saw the replay's CPU time (15 ms in New Donk City) as GPU time (4.5 ms there) and its dynamic resolution rendered
  the scene far smaller. They now come from a clock running at a quarter of real time (`SMO_VK_REPORT_TIME_SCALE`, default
  0.25; 1.0 as before), and the scene stays sharp.

Measured with `SMO_HITCH=25` on the same scripted 4-minute route (a new game's opening, Ryzen 5 3600XT, RTX 3090):
frames that missed the 60 Hz refresh by more than 25 ms, 173 without interpolation, 91 with it (and 78 of those bridged on
screen by predicted frames). The heaviest places (New Donk City's streets) are still bound by the render thread's replay
(about 15 ms a frame there) on a 6-core Zen 2.

## Repository layout

```
runtime/          the host runtime, by subsystem: core, libc, os, fs, gfx, audio, input, services, game
scripts/          setup.py (from your dump to a build) and run.py (start the game, play an input sequence)
tools/target.py   where the build's files go
tools/extract/    the import of your dump (NSO to ELF, version checks) and the function lists
tools/lifter/     the AArch64 to C lifter, the SDK lifter, the differential test and the build steps
tools/dev/        lookups for debugging: disassembly, callers, archives, stage placement, save warps, profiles
tests/            scripted Cap Kingdom run, stage stress test, every shader stage offline, translator equivalence,
                  texture layout, fast paths
docs/             building, architecture, debugging, scripted runs
```

## Plans

- Speed on slower CPUs and GPUs, and shorter stalls the first time a kingdom is visited (its textures and shaders are
  made then).
- Scripted runs further through the game: the rest of Cap Kingdom, then kingdom by kingdom.
- Checks on real hardware for the few shader instructions translated from documentation only (`SHFL`, `AL2P`, indexed
  `IPA`).
- Other physical controllers (PlayStation, Switch Pro) and other GPUs (AMD, Intel).
- What the 1.4.1 update adds: Balloon World, the Snapshot Mode filters, the Labo VR mode.

## Contributing

Testers with other GPUs and controllers, and people who know the game's internals, are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) first: nothing derived from the game goes into the repository, sources must be
clean, and every claim is measured.

## Related work

- [ssbu-recomp](https://github.com/Denkishi/ssbu-recomp): the static recompilation of *Super Smash Bros. Ultimate* whose
  design, lifter and runtime this project started from (GPL-3.0).
- [OdysseyDecomp](https://github.com/MonsterDruide1/OdysseyDecomp): the decompilation project of *Super Mario Odyssey*,
  read as a reference only; none of its code is included.
- yuzu/suyu and Ryujinx: documentation and sources consulted for console behaviour, cited where used.

## Legal

This project is not affiliated with, endorsed or sponsored by Nintendo. *Super Mario Odyssey* and *Nintendo Switch*
are trademarks of Nintendo and are named only to say what the software works with. The repository contains only
original source code and documentation (GPL-3.0-or-later, see [LICENSE](LICENSE)): no game code, binaries, assets,
console keys or firmware, and it does not help anyone obtain them. Function names and addresses in comments identify
locations in the game's own executable, for interoperability; they are not game code. You must own the game and dump
it yourself.

## Credits

- [ssbu-recomp](https://github.com/Denkishi/ssbu-recomp) by Denkishi: the design and runtime this project builds on.
- [Unicorn](https://www.unicorn-engine.org/) and [Capstone](https://www.capstone-engine.org/): reference CPU emulation
  and disassembly.
- [SDL3](https://github.com/libsdl-org/SDL), [glslang/shaderc](https://github.com/KhronosGroup/glslang), the Vulkan
  SDK.
