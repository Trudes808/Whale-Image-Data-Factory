# Whale-Image-Data-Factory
repo for creating whale training data images via unreal engine

Synthetic, pixel-labeled whale images rendered in Unreal Engine, for fine-tuning DINOv3
semantic segmentation. See [PLAN.md](PLAN.md) for the design and roadmap.

## Layout

```text
factory/            pure-Python code outside Unreal: disk guards, stand-in whale mesh, label QA
tools/              shell entry points: install, launch, build scene, render, disk report
unreal/WhaleFactory Unreal project (Content/Python holds editor-side scripts)
tests/              python3 -m unittest discover -s tests -t .
data/               rendered datasets (gitignored)
```

## One-time setup

**Git.** Do this after cloning. git-lfs must be on PATH; the user-local binary is in
`~/.local/bin/git-lfs`.

```bash
git config core.hooksPath .githooks   # LFS hooks + the large-file/licensed-asset guard
git lfs install --local
```

**Unreal Engine** (precompiled Linux build; no sudo needed). The engine is installed on the
root disk under `/var/tmp/sat3737-unreal/`, not on the shared, nearly full `/home`.

1. Start a FastX **Ubuntu Desktop** (or **Gnome**) session. In Firefox there, log in at
   <https://www.unrealengine.com/linux> and download `Linux_Unreal_Engine_5.8.3.zip`
   (match the 5.8.3 source clone). It goes to `~/Downloads`; snap Firefox can't write to
   `/var/tmp`.
2. Install it. This checks space, extracts to `/var/tmp/sat3737-unreal/UE_5.8.3`, and
   points `current` at it. `--delete-zip` then frees the zip's space on `/home`:
   ```bash
   cd ~/NOAA_whales/Whale-Image-Data-Factory
   tools/install_unreal.sh ~/Downloads/Linux_Unreal_Engine_5.8.3.zip --delete-zip
   ```
3. Headless sanity check (no GPU or display needed):
   ```bash
   tools/check_unreal.sh
   ```
4. Open the editor inside FastX. Use Activities → search **"WhaleFactory Unreal Editor"**,
   or run `tools/launch_editor.sh` from a terminal in the session. The menu entry is
   added by `tools/install_desktop_launcher.sh` (already done on this machine). It opens
   a terminal next to the editor where disk-watchdog warnings appear.
   On first open, Unreal may ask to associate or convert the project to the installed
   engine version. Accept, since the project has no assets yet.

To use a different engine location, set `WF_UE_ROOT=...` in `tools/ue_env.local.sh`
(gitignored).

## Whale model

The whale is the CC-BY "High-Poly Humbpack Whale" by K9239 (Sketchfab); see
[THIRD_PARTY.md](THIRD_PARTY.md) for the download location, the required credit, and what
the model is (a 16 m static mesh, no rig). Once the zip is extracted to
`third_party_assets/humpback_k9239/src/`, convert it for Unreal:

```bash
python3 -m factory.humpback_asset third_party_assets/humpback_k9239/src third_party_assets/humpback_k9239/converted
```

`build_scene` below uses it automatically. `WF_WHALE=procedural` falls back to the
stand-in mesh from `factory/procedural_whale.py`.

## Building and rendering the scene

Close the project in the FastX editor first; these run their own headless editor.

```bash
tools/run_editor_script.sh build_scene         # ocean, sky, whale, label material, cameras
tools/run_editor_script.sh make_render_config  # Movie Render Queue config (RGB + label pass)
tools/render_sequence.sh                       # -> unreal/WhaleFactory/Saved/Previews/mrq
python3 -m factory.qa overlay RGB.png MASK.png OUT.png   # visual check of a label
```

## Posing the whale at the surface

The mesh is rigid, so a pose is a rotation plus a height: `factory/poses.py` rotates the
mesh and sets its height so the highest point is a chosen clearance above calm water, and
places a boat camera to frame it. Poses: `logging`, `blow`, `arch`, `fluke_up`, `breach`,
`pec_up`, `spyhop`, `awash`, `submerged` (a negative).

White water (`factory/splash.py`) is calculated geometry: for impacts (breach, fluke and
pec slaps) a ragged cluster of water plumes, each thrown on its own ballistic arc from
where the body meets the surface, with droplets and foam patches; for surfacing poses a
lighter wash. Size and amount vary per shot (`splash` in the config). The meshes render
with a translucent white-water material whose alpha is noise torn toward the top and
sides of each sheet, and each carries a hidden coverage twin that writes custom stencil
2, so dense white water in front of the whale labels as background, the same way a boat
would. Thin spray (alpha under 0.35) does not cover.

```bash
python3 -m factory.poses third_party_assets/humpback_k9239/converted/SM_Humpback.obj -o data/shots/poses
tools/render_shot.sh data/shots/renders data/shots/poses/*.json   # each -> RGB, mask, overlay.png
```

Sea state is a preset added to any shot (`factory/seastate.py`): `glassy`, `calm`,
`light` (default), `moderate` (about 1 m waves), `rough` (2-3 m waves). The dataset
config weights moderate and rough at 55 percent of shots together.
It sets the wave spectrum, the surface detail and foam parameters, and wind-spray sprites:

```bash
python3 -m factory.poses third_party_assets/humpback_k9239/converted/SM_Humpback.obj logging breach --sea rough -o data/shots/sea
tools/render_shot.sh data/shots/renders/sea data/shots/sea/*.json
```

A shot JSON can also carry an `ocean` section (material parameters of
`/Game/WhaleFactory/Water/MI_OceanDeep`), `sun` and `fog`; `wf_shot.py` documents the
keys. Water defaults are in `wf_build_scene.py`: `OCEAN_LOOK` (absorption, scattering,
albedo, surface detail) and `OCEAN_WAVES_PARAMS` (the Gerstner spectrum, written into
`/Game/WhaleFactory/Water/WW_Ocean` at build time; `WF_OCEAN_WAVES_JSON=file` overrides
it for one build, `WF_OCEAN_WAVES=preset` keeps the plugin's default swell).

The label pass evaluates the same waves the water renders, so sea state can change
freely within one limit: keep the wave direction spread at 120 degrees or more (see
`factory/seastate.py`). `tools/run_editor_script.sh probes` adds striped 1 m poles around
the whale for checking the rendered water level; `WF_PROBES=clear` removes them.

`tools/render_sequence.sh /Game/WhaleFactory/Sequences/LS_TopDebug <dir>` renders a
top-down view, useful for checking water coverage. To see the scene in the editor, open
`/Game/WhaleFactory/Maps/OceanPOC`.

## Disk safety

`/home` is shared and nearly full. All tools here run Unreal under a watchdog.
They refuse to start below 30 GiB free, warn below 40 GiB, and terminate below 15 GiB.
Override with `WF_MIN_FREE_GIB`, `WF_WARN_FREE_GIB` and `WF_KILL_FREE_GIB`. To see
where space is going:

```bash
tools/disk_report.sh
```

Launch Unreal through `tools/launch_editor.sh`, not the raw binary. The raw binary skips
the watchdog, the pinned cache location, and the core-dump limit.

## Third-party assets

This repo is public. Put licensed assets (Fab, Sketchfab, and so on) under
`unreal/WhaleFactory/Content/ThirdParty/`. That folder is gitignored, and the pre-commit
hook also blocks it.

## Body deformation and skins

`factory/deform.py` bends the rigid mesh per shot (arch, tail curl, head pitch, lateral
stroke, fluke twist, pectoral fin raise and sweep), sampled by pose category; the
parameters are stored in the shot and in `meta/<id>.json` (`deform`), and the deformed
mesh is regenerated from them in post-processing so body coordinates and keypoints stay
exact. `factory/skin.py` makes skin variants from the base colour texture:

```bash
python3 -m factory.skin third_party_assets/humpback_k9239/converted/T_Humpback_BaseColor.png third_party_assets/humpback_k9239/skins --count 6 --seed 1
tools/run_editor_script.sh build_scene      # imports them as MI_HumpbackSkin_<k>
```

Each shot records its `skin`; the `body` section of the config sets the deformation
fraction and strength and the skin weights. In a batch every shot gets its own whale
actor (deformed mesh, skin), shown only in its frames.

Note for run01 (rendered before 2026-10-11): Unreal's OBJ importer mirrors Y, which the
symmetric base whale hid. Pixel masks are unaffected, but run01's `bodycoords` angle
channel, `keypoints` and `parts` have left and right swapped relative to the image. Later
runs are correct (all OBJ writers negate Y).

## Building a dataset

```bash
tools/build_dataset.sh data/renders/run01 2000 1       # shot list -> batches -> renders -> dataset
nohup setsid tools/build_dataset.sh data/renders/run01 2000 1 > data/renders/run01.log 2>&1 &
```

`factory/shotlist.py` samples shots from `factory/config/poc_boat.json` with a seed: pose
category with continuous pitch/roll/yaw/height jitter, camera bearing all around the
whale, distance, boat or elevated height, focal length from how much of the frame the
whale should fill, aim offset and roll, sea state, water colour, sun elevation and
azimuth, overcast or clear, haze, exposure, spray, and negatives. From run02 the camera
also enforces a minimum on-screen size: the above-water part of the whale must span at
least `camera.min_extent_frame_fraction` of the frame (0.25) with
`camera.min_in_frame_fraction` of it in view; the lens is zoomed in by the shortfall (a
photographer frames what is visible) and the draw repeated up to `max_camera_tries`.
`camera.extent_frame_fraction` / `in_frame_fraction` are recorded per shot. The run01
config (whales down to 3 % of the frame, more submerged negatives) is kept as
`factory/config/poc_boat.run01.json`. Shots are grouped into
batches by sea state and water colour; `tools/render_batch.sh` keys each batch into one
level sequence (`wf_build_batch.py`) and renders it in one game session, about 2 s per
shot plus editor start-up per batch. `factory/postprocess.py` then writes:

```text
data/renders/run01/dataset/
  images/<id>.png     RGB
  masks/<id>.png      palette PNG, pixel values 0 background, 1 whale (above water, not
                      spray), 2 whale just below the surface (from run02: whale pixels not in
                      the rendered label whose surface is within labels.through_water_depth_m
                      of the mean water plane, so the body may show through the water; an
                      "ignore" class for segmentation, run01 has none); the palette shows 1 as
                      white and 2 as grey in viewers, and np.asarray(Image.open(p)) gives the
                      class ids (cv2 expands the palette to colours: 255 / 128)
  amodal/<id>.png     body-part id of the whole whale ignoring water (fluke, peduncle,
                      dorsal_back, mid_body, head), from projecting the mesh
  bodycoords/<id>.png canonical body coordinates per pixel of the amodal whale: R = fraction
                      along the body (tail 0 -> snout 255), G = angle around the body axis
                      (0 back, 64 right flank, 128 belly, 192 left flank), B = 255 where
                      defined. The same body location maps to the same value in every photo.
  depth/<id>.png      uint16 depth (cm) of the whale surface
  meta/<id>.json      everything about the shot: pose, camera, sea state, lighting, the
                      whale's orientation relative to the camera (relative bearing, view
                      elevation, distance, pitch/roll), the full 6-DoF pose in the camera
                      frame, camera intrinsics and pixels per metre at the whale, named
                      keypoints (snout, blowhole area via landmarks file, dorsal fin tip,
                      fluke notch and tips, pectoral tips, eyes) with pixel position, depth,
                      occlusion and whether they are inside the label, visible fraction of the
                      whale and of each body part, bounding box, QA flags
  landmarks_mesh_frame_cm.json  the keypoints' positions on the mesh
  manifest.jsonl      one line per shot
  splits.json         train/val by sea-state/look group (factory/dataset.py)
  summary.json        distributions
```

The orientation and part fields exist for the follow-up task (which part of the whale
is seen, and from where). The projection uses the CineCamera's 23.76 mm filmback,
calibrated against rendered masks (`factory/project.py`).

## run03: body-coordinate / pose training data (2026-10-12)

`factory/config/run03_bodycoords.json` (build with `tools/build_dataset.sh RUN_DIR COUNT SEED factory/config/run03_bodycoords.json`):

* Framing matched to long-lens field photos: per shot a target for how much of the frame the
  *visible* whale spans (`camera.extent_target`, log-uniform 0.3–1.6; the lens zooms to it, the whale
  may be cut by the frame edges, `min_in_frame_fraction` 0.5), distance 40–600 m, focal 50–1500 mm.
* Close-up and rolled categories (`factory/poses.py`): blow_sliver, fin_tip, fluke_edge, peduncle,
  spyhop_low, roll_belly; larger pitch/roll jitter; deformation on 95 % of shots at 0.7–1.8 strength;
  body length scale 0.55–1.1 (`body.length_scale`, shot `body_scale`; the canonical labels stay on the
  unit mesh, the rendered mesh is scaled in the per-shot OBJ).
* Labels per shot: 18 keypoints (`project.landmarks`: snout, chin, blowhole, dorsal fin tip/front/back,
  tail base top/bottom, fluke notch/tips, pectoral tips/bases, belly, eyes) with pixel position, depth,
  whale-frame and canonical 3D position and a `state` (visible / under_water / hidden / occluded /
  out_of_frame, from the depth buffer, the rendered mask and the mean water plane); `water_row_px`;
  `canon/` (point on the undeformed unit mesh, for correspondence) and `canon_deformed/` (point on the
  posed, bent, scaled mesh, for pose fitting); `deform` parameters and `body_scale` in meta.
* `python3 -m factory.qa_sheet RUN_DIR` draws the keypoint states, water row and posed mesh on a
  contact sheet for label QA.
* Lessons from the smoke batches (2026-10-12): keep the camera within ~180 m (the water mesh LOD
  range; beyond it the rendered surface and the label pass disagree and a visible back gets an
  empty mask); choose the sea state before placing the whale and lift low poses by part of the
  crest height (`body.wave_clearance_fraction`, `seastate.crest_height_m`), otherwise slivers in
  rough seas are washed over; keep splash meshes rare and small for close-ups (they render as large
  white sheets near the camera); keypoint visibility uses a 5x5 neighbourhood of the mask.
* Water physics (2026-10-12/13, after the smoke review): the cine camera's default lens clamps the
  focal length at 1000 mm (`wf_shot.unclamp_lens`; above it the render was smaller than the labels,
  caught by the `projection_not_covered` QA flag); each shot has a frozen wave time (`wave_time_s`,
  a console-variable track on `r.Water.OverrideWavesTime`) and `wf_build_batch.ride_waves` lifts
  the whale, camera and splash onto the Gerstner surface at that time (`water_surface_z_cm` in
  meta, the batch as rendered in `raw/<batch>/batch.json`); depth of field like a field lens
  (`camera.aperture_f`, focus on the whale's above-water centroid) softens the sharp wave bands
  that resolved Gerstner chop shows under a long lens at grazing angles (supersampling and water
  roughness did not); spray droplets are 2-10 cm (28 cm ones read as confetti at distance).
* run04 to-do (from the run03 review): white water along the whale's flanks is far from real photos. A
  diving or breathing whale pushes a bow wave and wash along the body-water contact, and the run03
  segmentation model learns to include that turbulence in the whale (backs and tails). The splash
  model should put foam and sub-surface turbulence along the contact line (not only crown/wash blobs
  about one centre), with foam textures, and label it as background. Also: a bluer, darker horizon so
  the wave bands lose contrast under long lenses; a veiled submerged body in clear water; some land
  (islands, horizon features) in the backgrounds so the segmentation stops firing on them.

## run04: realism, the rigged whale, individuals, mom + calf (2026-10-13)

Config `factory/config/run04_realism.json`; end to end with `tools/run04_pipeline.sh` (render → qa →
`Dinov3-WhaleID/scripts/run04_train.sh` → `run04_evaluate.sh`, report `experiments/run04_report.md`
against the run03 fine-tunes).

* **Whale model**: the purchased rigged 4k humpback (`third_party_assets/humpback_4k`, glTF, 25 joints,
  one 29 s dive-then-breach clip). `factory/humpback4k.py` converts it (OBJ + textures, project frame,
  cm); `factory/skinned.py` is the glTF loader + linear blend skinning; `factory/whalemodel.py`
  (`WhaleModel`, config `"whale": {"kind": "humpback4k"}`) poses it per shot: a clip time `pose.clip_t`
  sampled from per-category windows (derived from the spine joints: straight / arch / tail up /
  head up; the bind pose itself is arched and is never used as "straight"), the front half levelled,
  then the category's pitch/roll/yaw as before. Every rigged shot gets its posed OBJ
  (`deform.write_batch_meshes`); postprocess/qa use `WhaleModel.posed`. The labels stay in the K9239
  canonical frame: the rest pose is registered slice-wise to the K9239 mesh (`register_to`), so body
  coordinates and the 18 keypoints keep their meaning (`WhaleModel.landmark_indices`: geometric picks
  plus rig joints for the pectorals and the K9239 eyes mapped through the canonical frame). The
  file's two "jingxu" (baleen) block meshes are dropped. In the editor `wf_shot.ensure_humpback4k_materials`
  builds the skin material (same graph as the K9239 one: BaseColor parameter, RoughnessScale for the
  wetness track, normal map) and one instance per skin texture, lazily, and rebuilds them when the
  source textures change (stamp in `Saved/wf_humpback4k_import.json`).
* **Throat colour**: the purchased texture paints the underside of the head and the chest pink, which reads as
  an open red mouth whenever the chin shows (breaches, spyhops, rolled whales). `humpback4k.neutralise_throat`
  turns those texels into the pale pleated ventral white (shading kept); the untouched original is
  `converted/T_Humpback4k_BaseColor_orig.png`. Every individual texture derives from the corrected base.
* **Individuals** (`factory/individuals.py`, `python3 -m factory.individuals --count 12 --calves 4`):
  copies of the model with their own proportions (rostrum, girth, tail stock, fluke span, pectoral
  length, dorsal fin height 0.7-1.6x and sweep: the most variable feature on a real whale's top contour), resting joint angles around the keypoints (fluke pitch, tail stock, head, pectorals), fluke
  outline (sweep, notch depth, trailing-edge wear and nicks) and colouring (a `factory/skin.py`
  variant plus a Katona class 1–5 fluke underside painted on the UV island). Library
  `third_party_assets/humpback_4k/individuals.json`, textures `skins/T_Humpback_BaseColor_<id>.png`;
  a shot carries `individual` and `skin` = the id. Vertex i stays canonical vertex i.
* **Mom + calf** (`companion` in the config): a calf individual (`c..`, scale 0.35–0.5, chubbier) beside
  the mother on 18 % of the surface shots, roughly her heading, placed off her flank on the wave
  surface; the camera frames both. It renders with custom stencil 3 → 0.6 in the label pass
  (`M_WhaleVisibleMask`; after a shader change rebuild it with `tools/run_editor_script.sh rebuild_labels`).
  `masks/` keeps every whale as class 1 (segmentation), `instances/<id>.png` tells 1 = the shot's whale,
  2 = the calf, and the mother's amodal/body-coordinate labels are cleared under the calf. The
  Dinov3 loaders crop on the primary instance.
* **White water**: the foam carpet has lumpy thickness (up to ~10 cm where dense), the material carves holes from the foam texture, and stronger
  break-up, so it reads as a raised rim at grazing angles instead of a flat line; empty carpets are
  skipped. Known open point: long-lens wave bands (see run03) are unchanged by the model switch.
* **Poses on the rig** (after the first test renders): the clip is sampled at hand-picked keyframes per
  category (`whalemodel.KEYFRAMES`), never at arbitrary frames of the breach sequence; the tail-up categories
  take most of the frame's own fluke lift out of their nose-down pitch. `back_arch` is the commonest field
  shot (breath, then the back and dorsal fin rolling through with head and flukes under): a straight keyframe
  plus the procedural hump (`arch_deg` -30..-80, apex `arch_centre` at the dorsal fin, head and tail stock
  curled down), clearance measured on the back only. The dive clip only ever lifts the fluke blade (+4 to
  +73 deg through the dive), so `fluke_up` adds a per-shot `fluke_curl_deg` at the fluke base (vertex level,
  levelled frame): tips hanging down early in the dive (clip_t <= 1.2, curl -70..-10) as the whale doubles
  over, tips up later; `fluke_edge` always curls up (0..25); `fluke_down` (the start of a fluking dive as the
  field photos show it, e.g. 184A2635: tail stock arched high and clear, blade still hanging down, tips at the
  water) uses the lifted-tail frames with a -95..-60 curl and a clearance cap of 1.8 m. In the three fluking
  categories the head is never out of the water: the pitch stays at or below -15 deg and the nose is lowered
  until the front body is submerged (`shotlist.HEAD_UNDER_CATEGORIES`). Joint offsets (individuals) rotate about the
  whale's own axes at the frame (`WhaleModel.body_frame_offsets`), not world axes. Pectoral angle is a per-shot `pec_droop_deg` applied at
  vertex level in the levelled frame with a mesh-smoothed fin membership (surface whales carry the fins
  down, the rolled categories raise one). None of the K9239 procedural parameters (fin rotation, twist,
  lateral) apply to the rig: its fin heuristic tears this mesh at the shoulder. Individual edits are made in
  the skinned rest frame and mapped back through each vertex's blended skin matrix
  (`Rig.bind_from_rest`); a seam stitch closes the few centimetre-scale edges the asset's own skinning
  overstretches. An edge-stretch tear check over all categories (max ~2x on tiny edges) is the acceptance
  test for any new pose code.
* **Contour classes** (`contours/<id>.png`, palette): the visible mask's boundary labelled from the render's
  own answer about the water (the whole-whale projection minus the visible mask = body hidden by the surface,
  by crests in front of the whale or by foam): 2 = the seam into water / foam (the body continues hidden past
  the edge), 3 = edge against the calf, 1 = everything else: an edge in open air, the whale's own outline
  against sky or open water. The frame edge of a whale cut by the picture is no contour at all (neither
  class, since 2026-10-14). `visible.contour_true_fraction` in the metadata. Meant for contour-weighted
  segmentation / visibility losses (Dinov3-WhaleID `common/contours.py`, run04 training) and for a silhouette
  fit that matches the top contour and only forbids mesh above it, never matching the waterline. Review
  sheet: `contours_hidden_sheet.png` (hidden body tinted red).
* **Frozen wave time (fixed for run04)**: the per-shot `r.Water.OverrideWavesTime` cvar track never reached
  the water (probes of the label shader showed the surface following the sequence clock: 1.17 s per shot,
  not the shot's time; the cvar is ECVF_Cheat, so Engine.ini cannot set it either). Every render now freezes
  the ocean at t = 0 through a startup console command on the render process (`tools/render_sequence.sh`,
  `-ExecCmds="r.Water.OverrideWavesTime 0"`; probe-verified: the label shader reads the frozen surface on
  every frame), and the shot sampler varies the wave phase by translating the whole scene (whale,
  camera, calf, splash, white water; `scene_offset_cm`, up to +-150 m) across the frozen wave field,
  choosing among candidate offsets the one where the surface is most even along the body (the whale rides
  the swell). `tools/water_check.py` must report IoU >= ~0.9 on a water-only render before any run.
* **Wave seed bug (fixed for run04)**: the ocean renders one wave set per batch, but up to run03 every shot had
  its own wave seed, so for every shot but the first of each batch the lift onto the surface, the white water
  and the through-water labels were computed on waves the render never showed. The seed is now a function of
  (sea state, look), the batch key; phase variety comes from the scene offset.
* **Two render stops in run04 (2026-10-14/15)**: at batch 212 the headless editor died seconds into startup (an
  engine-internal assertion in its HTTP client thread, `FCurlHttpConnectionPool`; transient). At batch 220 the
  editor never launched: the Zen derived-data cache (`WF_ZEN_DATA_DIR`, on the root disk) had grown to 114 GB
  over 220 batches (every batch adds generated splash / white-water meshes, ~0.5 GB of derived data per batch)
  and `factory.diskguard` refused to start the editor below its 30 GiB floor, while the step's stale log from
  the previous batch still read "OK". Fixes: `tools/run_editor_script.sh` deletes the previous log before
  launching and fails when none is written; `tools/render_batch.sh` retries the build and the render step once
  each; and the Zen cache (`/var/tmp/sat3737-unreal/ZenData`, `cache/` + `cas/`) is safe to delete between runs
  when no `zenserver` process is up (the editor rebuilds it; the first batch afterwards recompiles shaders).
  Check `df /` as well as `df /home` before a long run.
* **QA flag `projection_not_covered` (fixed 2026-10-14)**: the check that the rendered mask covers the projection's
  above-water part used the image row of the waterline, which is no test of being above water once the camera is
  elevated or the whale arched: the submerged far body projects above that row, and 14 % of calm-sea shots of the
  first run04 batches were flagged (and would have been dropped from training) with correct labels. It now uses
  the per-pixel height of the whale surface above the local frozen Gerstner surface (> 25 cm counts as clearly
  above water) and flags only when less than half of that is covered: what stays uncovered legitimately is body
  behind crests or foam, so the flag marks a render / label mismatch (e.g. a clamped lens), not partial occlusion.
  `projection_covered_by_mask` in the metadata carries the fraction either way.
* **Camera bearing** (`camera.bearing_sectors`, [start, end, weight] degrees about the whale; 0 = ahead of the
  nose, 180 = behind): run04 draws 45 % of cameras from behind (following), 20 % from each side (alongside) and
  15 % from ahead, the way the field photos are taken. The drawn bearing is in `camera.bearing_deg`.
* **Label pass**: `tools/run_editor_script.sh rebuild_labels` rebuilds `M_WhaleVisibleMask` (built under a
  temporary name and swapped in on success; the MRQ preset references it by path, so a missing material means
  renders without mask frames). Never run it while a render is going: two editors on the project lock each
  other out. Whales in a trough seen from a low camera in rough water are legitimately hidden behind crests
  (empty mask, flagged): the sampler now keeps the whale on the swell (best of 2-3 wave phases in moderate /
  rough seas) and boat cameras at >= 1.5 / 2 m there.
* Test renders: `data/renders/h4k_test*` (rigged model, individuals, calves, keyframes, back arch).

## run05: surfacing poses that show the back, and tighter framing (2026-10-16)

A comparison of run04 with the real photos of 2024-12-18 (Dinov3-WhaleID `experiments/run04_domain_check/`) found two
mismatches: a close-up tail the field never shows (20 % of shots with the whale over 20 % of the frame, lenses to
1500 mm), and surfacing poses that expose the head where real surfacing photos show the back from the peduncle to
the blowhole (the rostrum is 1 % of real whale pixels, 39 % of run04's). The rig's "straight" keyframes sit nose-up
in the levelled frame (at zero bend the snout is 70 cm above the dorsal fin tip), so a mild hump centred mid-body
lifts the blowhole-to-head region; the recipe that gives the real mix is the back_arch one, a hump with its apex at
the dorsal fin (body fraction 0.36-0.42) and the head pitched well down. `factory/config/run05_realism.json`:
* `surface` (new, weight 30) and the breathing / resting categories (`logging`, `blow`, `blow_sliver`, `awash`) are
  procedural back-apex poses (`shotlist.PROCEDURAL_POSES`, overridable per config in `procedural_poses`), with the
  clearance measured on the back only (`poses.BACK_HEIGHT_CATEGORIES`, now without the head: `BACK_X_MAX_CM`), small
  per-category pitch / roll jitter (`pose_jitter.pitch_deg_by_category`, `roll_deg_by_category`) and low clearance
  caps (sliver 0.45 m, awash 0.3 m). Test renders: `r05_test3` (surface mix [0, 0.37, 0.43, 0.19, 0.01] tail->snout,
  head part 7 % visible), `r05_test4` (the four breathing categories).
* framing: `extent_target` 0.08-0.7, focal 70-600 mm, distance 50-400 m.

## run05 prep: the Niagara splash pack (`Niagra_splash_pack/`, not yet in the project)

A Fab pack of baked fluid simulations played back as vertex-animation-texture meshes through Niagara:
`NS_Splash_Small / Mid / Mid_Angle / XL`, `NS_Breach_Small / Mid / Mid_Angle` (each with Main water,
white-water and mist layers: `SM_*_Main/WW/Mist_mod` meshes driven by `T_*_pos/rot` textures), plus a
ripple disk with a foam-surface material. Package versions 1004-1012 (ours 1018): loads in 5.8.3.
What it covers: impact events (breach landings, fluke and pec slaps, big splashes). What it does not:
wash along a surfacing back, bow wave / wake, foam trails, i.e. the white water the field photos show on
most frames (and the segmentation's main confusion). Integration plan for run05, after run04 is judged:
1. copy `data/Content/SplashPack` into `Content/ThirdParty/SplashPack`; 2. drive the VAT materials
directly from a StaticMeshActor with a per-shot frame parameter (the materials take the frame through a
Niagara dynamic parameter: expose it as a scalar parameter in an instance, or spawn the Niagara system and
key its age in the sequence), so the splash is a deterministic mesh the label pass can see; 3. a hidden
stencil-2 twin with the pack's WW mesh for coverage, as for the current splash; 4. keep `factory/whitewater`
for the along-body wash and wake. Expected gain: breach / slap realism and more convincing "spray is not
whale" examples; little effect on the common surfacing shots.
