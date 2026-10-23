# Plan: the render thread's replay in New Donk City (and the cars)

State on 2026-10-22 (branch `smo-1.3.0-interpolation`, commit `dbdb78d`): New Donk City's streets run at 44-54 fps
(Ryzen 5 3600XT, RTX 3090). Per frame there: the game's main thread about 8 ms, the GPU about 4.5 ms, the render thread's
replay of the frame about 15 ms (draw preparation 8-9 ms for about 2,000 draws, about 22 MB of guest memory copied into
the stream ring, the driver about 2 ms). The replay is the limit. Reported by the player: the frame rate falls hard
whenever a car is in view.

Benchmark used throughout (about 1 min 50 s a run, one at a time, never two games at once):
- save: `build\save_city_street` (made with `tools\dev\save_warp.py build\save_city --out build\save_city_street
  --stage CityWorldHomeStage --visited --checkpoint obj4904`); copy its files into `<run folder>\save\` (a run's
  `--userdata` is a user-data root, the save goes in its `save\` subfolder)
- sequence `city_short.seq`: `@until-nerve:StageSceneNrvPlay:Z/30`, `120`, `1:u/600`, `1:r/400`, `1:d/600`
- `scripts\run.py <name> --seconds 110 --seq city_short.seq --userdata <folder> --env SMO_SEQ_EXIT=1 --env SMO_VK_INTERP=1`
- read: `nvn_vk: frames A..B: N fps, D draws per frame`, `nvn_vk_perf:` (batch replay, draw preparation, wait for the
  GPU), `nvn_perf: cpu ms/frame by thread`; `SMO_HOSTPROF=35,12,2` for the two busiest threads,
  `tools\dev\profnames.py`; `SMO_VK_GPUTIME=1` for GPU time.
- runs of the same build differ by several fps (traffic, crowd): judge a change by two runs, or by the per-frame
  render-thread numbers rather than fps alone.
- every change: `tests\run_cap_kingdom.py <name>` must still PASS (14 of 14, 0 host exceptions).

## Step 0: the cars (first, cheap)

1. A sequence that keeps traffic in view: from the street checkpoint, turn the camera towards the road and stand
   (`1:6/120` then `600`), next to one that looks away (`1:4/120` then `600`). Compare `nvn_vk_perf:` and the hostprof of
   the render thread between the two.
2. Suspects, in order: (a) the compute copy-backs: `nvn_vk: compute at frame ...` counts copy-backs forced inside a
   batch (11,819 in one city run) and waits for the GPU before a copy-back (569, 1.4 s in all): if cars are drawn from
   compute output (skinning, deformation) each such wait stops the replay for a GPU round trip; (b) conditional renders
   on occlusion results (`SMO_VK_OCC_LOG=1` logs the ones that wait); (c) a draw kind that goes to a slow path
   (`SMO_VK_VTXCHECK=1`, texture made or read at its use: `nvn_vk_upl` lines with `SMO_VK_HOLDLOG`).
3. Fix what it is: for (a), keep the compute output on the GPU and let the draws read it there (a device buffer bound
   as the vertex stream) instead of copying it back to guest memory and streaming it again, wherever the CPU never reads
   it; only copy back when the game reads the range.

## Step 1: cheap cuts in the replay (each measured alone)

- `memcpy` / `ring_put` (about 2-3 ms): the 22 MB a frame are guest buffers copied into the ring. Import the game's
  memory heap into Vulkan once (`VK_EXT_external_memory_host`, the GPU has it) and bind guest buffers in place; copy only
  ranges written during the batch (the existing write log, `host_wrote`). Biggest single cut, moderate risk (alignment of
  the import: host page size; lifetime: the heap is one fixed reservation).
- descriptor sets: 2,400 allocated a frame, 1,600 reused; cache texture sets across frames by their binding list hash
  (the hash already exists) instead of per command buffer.
- `resolve_sampler`, `scan_list`, `pipe_get`, `nvn_front_handle`: look at each with the profile; small hash tables
  (`nvn_label_get` 1,024 buckets, `nobj` 4,096) to 65,536.

## Step 2: replay on several threads

The replay walks the frame's NVN command lists in order and changes one state (`R`): stream ring position, descriptor
pools of the command buffer slot, the pipeline bound, textures made at first use, occlusion segments, copy-backs. So:

1. **Split points.** Cut the frame at render pass boundaries (`pass_end`). Within a pass the draws only read state set
   by the commands before them. Mark the commands that must stay in order on the render thread: compute dispatches,
   copy-backs (`cb_sync`), occlusion report / reset / conditional render, texture creation and uploads, buffer copies and
   clears, fences, barriers.
2. **Two-stage pipeline first** (lower risk, likely 30-40 % of the replay): a worker thread decodes the next pass's
   commands (handle lookups, vertex layout, pipeline key, texture and sampler resolution, ring copies into a
   per-worker ring section) into a compact draw list while the render thread records the current pass's `vkCmd*` from
   the decoded list. Needs: a per-worker sub-ring of the stream ring (fixed slices), thread-safe texture lookup
   (`vt_get2` takes a lock or creation is deferred to the render thread), the state snapshot at each split point.
3. **Parallel recording** (later, if step 2.2 is not enough): passes recorded into secondary command buffers
   (`VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT` with dynamic rendering's inheritance info) by 2-3 workers, each
   with its own descriptor pools and ring slice, executed in order by the render thread.
4. Behind a switch, `SMO_VK_MT=0/1`, off until checked: pictures identical to the single-threaded replay
   (`SMO_SHOT_EVERY` captures of the same scripted run, compared with `tools\dev\drawdiff.py`), `run_cap_kingdom.py`
   PASS, the stage stress test on a few kingdoms (`tests\stress_stages.py`), then on by default.
5. Threads: the workers go with the render thread on the processors away from the console cores
   (`os_pin_host_thread`); there are three such cores on a 6-core Ryzen, so at most two workers.

Target: the render thread under 12 ms a frame in the city, which with the asynchronous present gives 60 fps there.

## Design notes for Step 2 (from a read of the code, 2026-10-22)

- Unit of work = draws, not passes (passes are ended by the render thread itself: vt_layout, resolve_sampler feedback,
  tex_barrier, write_region, cmd_submit). A helper thread H walks the same command lists and turns each OP_DRAW_* into a
  decoded record (DDraw: ranges with ring offsets / CB index, samplers {Tex*, VTex*, view, sampler}, PKey + Pipe*, vertex
  input arrays) pushed into a single-producer single-consumer ring; the render thread RT keeps running exec() for every
  op and only the draw cases pop a record (draw_record). The NVN state fields of R move into an NState struct with an
  exec_state() both threads run.
- Stall ops (H waits until RT has run them): REPORT_VALUE, COUNTER_REPORT types 0 / 2-16, DISPATCH(_INDIRECT),
  COPY_BUF_BUF, UPDATE_UBO, CLEAR_BUF, COPY_TEX_BUF. Pass-through on RT: fences, occlusion reset/report, conditional
  render (RT drops records when render_off), barriers, texture copies.
- Ring: RT reserves 2 MB slices with ring_raw(NULL, SLICE) for H (wraps and ring_reuse waits stay on RT); slot
  bookkeeping lowered to the slice start at consume (ring_note_read); a host-write sequence log (g_hw, g_hw_seq,
  appended in host_wrote) lets RT re-copy ranges written after H copied them; CB epoch (g_cb_epoch) re-resolves ranges
  that used a copy-back buffer.
- Increments behind SMO_VK_MT: MT-0 the thread + scan_list on it; MT-1 buffer copies ahead (skip if external memory
  host import lands); MT-2 full decode with SMO_VK_MT_CHECK=1 comparing against the single-threaded result; MT-3 CBs,
  indirect draws, quads; MT-4 tuning. RT and H on different physical cores (SetThreadIdealProcessor).
- Cheaper single-threaded wins first: vstage_get slow path (byte-wise FNV over a whole shader when a vertex program
  alternates variants: memoise per Stage); sampler_get linear memcmp over up to 2048 samplers per sampler per draw (hash
  it); pipe_get (lock + byte-wise FNV + 128-byte memcmp chains: a direct-mapped Pipe* cache in front); word-wise hashes
  for the texture-set key and scan tag; skip vkCmdSetVertexInputEXT / topology / dynamic state calls that repeat the
  last values; passshot_frame() once per batch; index_max cache 2048 -> 16384.

## Next steps (2026-10-23)

State (after step 3 below): New Donk City streets 59-60 fps, render thread replay about 11.1 ms (draw preparation 5.6).
Before it: 17.9 ms a frame (about 56 fps), render thread replay about 12.7 ms (draw preparation 7.1).
60 fps needs 16.7 ms: about 1.2 ms more, plus a margin. Local branch has the four micro-optimizations of the worktree
agent cherry-picked (eb13d4f texture sets across command buffers, a088a83 index_max, 75c5a60 passshot flag, 8981476
word-wise hashes): push only after `tests\run_cap_kingdom.py capk_ag` PASSES (log build\run\capk_ag.log).

1. **Drops on entering the city** (frames 1200-1500 after the first play frame: p99 63 ms, max 218 ms; no pipeline
   holds). A hitch trace was started: `build\run\entry.log` (SMO_HITCH=40, SMO_VK_HOLDLOG=1). Read its `hitch: frame`
   lines and `nvn_vk_upl` lines: textures made or read at their first use (vt_get uploads on the render thread) are the
   first suspect; then page faults (heap growth) and the game's own loading threads. Fixes by cause: make textures ahead
   further (tex_ahead) or upload them from a worker into a transfer queue; pre-touch / pre-commit the guest heap.
2. **Occlusion wait at the present** (~1 ms a frame): measure `--env SMO_VK_OCC_PRESENT=0` (2 runs of city_short.seq);
   if no visible popping, make it the default.
3. **No ring copies of guest memory**: DONE for vertex and index data (`SMO_VK_HOSTMEM`, default on, commit 692b371).
   Measured first: 17.6 of the 20 MB a frame were vertex (15.6) and index (1.6) ranges, nearly all in CPU_UNCACHED pools
   (flags 0x22: no copy can be kept across batches; a vertex cache on the CPU_NO_ACCESS / CPU_CACHED rule of the textures
   would have covered 1.2 MB a frame). One import per memory pool costs about 45 ms in the driver and stalls the render
   thread even from another thread (211 pools entering the city: 19 fps), so the whole heap (3.1 GB, every pool is in it)
   is imported once at start in 14 overlapping pieces of 256 MB, on a thread of its own. Render-thread writes into guest
   memory wait first for the command buffers reading the range in place (hm_guard, 0 waits in the city). Result: replay
   13.9 -> 11.1 ms a frame in the streets (59-60 fps), 12.5 -> 9.8 ms entering. Uniform (2.5 MB) and storage ranges are
   still streamed (UpdateUniformBuffer rewrites uniform ranges mid-batch: in place they would need many guard waits).
4. **resolve_sampler cache** (~1-1.5 ms): cache (pool, handle) -> (VTex, view, sampler) per batch, dropped when the game
   writes the texture or sampler pool (host write log / texhash of the pool range).
5. Later: helper threads on the idle second hardware threads of the game's cores (CPUs 6, 8, 10) instead of the
   runtime's three cores; a newer clang with PGO for the runtime and the lifted code.
