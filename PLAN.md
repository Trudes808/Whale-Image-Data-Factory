# Whale Image Data Factory — Plan

## Goal

Generate labeled synthetic training data in Unreal Engine for fine-tuning DINOv3 to do
semantic segmentation of whales in field photos. The labeling rule:

- **whale** — whale pixels that are directly visible to the camera (above the water surface).
- **background** — everything else, including whale pixels hidden behind water (submerged)
  or behind any other object between the camera and the whale.

Proof of concept (POC): one swimming **humpback whale** in an open ocean, photographed from
boat-like viewpoints around it.

Later expansion: boats and islands as occluders and distractors, overhead drone views,
more species and behaviors, and multiple whales per frame.

Downstream consumer: `Dinov3-WhaleID/helpers/whale_isolation.py`, which currently uses
classical image processing (`dark_anchor_local_growth_v4`). The POC succeeds if a model
trained on this data beats that baseline on real photos.

## Feasibility Summary

Feasible. In a game engine the labels are the easy part: the renderer knows which object
is nearest to the camera at every pixel, so masks come out pixel-perfect at no cost.
The real risks are elsewhere:

1. **Sim-to-real gap.** Whale skin, wet sheen, white water and wake around the body,
   haze, and camera and lens character. Mitigations: domain randomization, strong
   training augmentation, DINOv3's robust features, and above all a small *real*
   labeled evaluation set (Phase 4).
2. **Getting Unreal running here.** This server is headless (SSH, no display), with
   about 144 GB free on `/home`. A source build of Unreal won't fit. Epic's precompiled
   Linux build will. Headless batch rendering (`-RenderOffscreen`, Vulkan) works on the
   NVIDIA RTX 4000 Ada. Scene authoring happens in the editor on this server over the
   existing FastX remote desktop session.
   Check early that the session gets hardware-accelerated Vulkan on the NVIDIA GPU
   (not llvmpipe/lavapipe).
3. **Whale asset quality.** A rigged humpback model with a swim cycle is needed.
   Start with a free one (Fab/Sketchfab) and upgrade to paid only if realism limits
   sim-to-real. Asset licenses may forbid committing the raw files to a public repo.

## Core Technique: How the Labels Are Computed

The whale mesh renders into the Custom Depth buffer with Custom Stencil = 1. A
post-process material (`M_WhaleVisibleMask`, built by `wf_build_scene.py`) computes, per
pixel:

```text
whale_visible = CustomStencil == 1
            AND CustomDepth <= SceneDepth + eps     # nothing opaque in front (boats, islands)
            AND camera ray to the whale point never dips below the wave surface
```

**What we learned in the spike.** The Water plugin's surface is *not* in `SceneDepth` as
seen by post-process materials. A debug depth pass showed the sea floor behind the water.
The water mesh also does not render into custom depth. So the water test is analytic:

1. Rebuild the whale point from custom depth.
2. March the camera ray, 128 steps, skipping samples above the highest possible crest.
3. At each sample, evaluate the **same** Gerstner waves the water renders with. These
   come from `GetAllGerstnerWavesNew` in `/Plugin/Water/Private/GerstnerWaveFunctions.ush`,
   using the ocean's water body index and the `MPC_Water.Time` parameter the water uses.
   Height at an XY point is found by inverting the horizontal Gerstner displacement
   (4 fixed-point iterations).
4. If any sample is underwater, the pixel is background.

Wave crests between the camera and the whale are handled. Measured against the rendered
RGB, the mask's waterline is within 1–4 px (median 1 px) at 1280×720.

- **Alignment.** The RGB image and the mask come from the *same* Movie Render Queue frame.
  The mask is an extra post-process pass, which guarantees pixel alignment.
- **Edges.** The label is computed per pixel from depth and stencil, so it is crisp
  (pure 0/255).
- **Spray, blow, and foam** must not be labeled whale (a splash labeled whale would teach
  the model that white water is skin texture or scarring). Splashes are calculated meshes
  (`factory/splash.py`: a ragged cluster of ballistic plumes + droplets + foam for impacts,
  foam patches and low plumes for wash; no blow spout for now), rendered with a translucent
  white-water material whose alpha is noise torn by height. Each carries a hidden masked
  twin that writes custom stencil 2 where the alpha exceeds 0.35; the label shader only
  accepts stencil 1, so dense white water in front of the whale is background. Thin mist
  still shows the whale through it and stays whale. A later refinement is an `ignore`
  (255) band for the semi-transparent fringe, and a Niagara upgrade for the look.
- **Extra output (to do).** Save the *amodal* whale mask too: stencil only, ignoring
  occlusion. That supports a later "submerged but visible" class for drone imagery.
- **Keep in sync.** If the waves get depth attenuation (shallow water) or the ocean
  height or index changes, the shader's `BASE_Z` / `BODY_INDEX` defines must change with
  it.
- **Wave spectrum limits.** With a narrow directional spread (`direction_angular_spread_deg`
  of about 50 or less) the plugin's water surface stops occluding anything below the calm
  level, so the whale and even test poles draw through the water while the label (which
  evaluates the waves correctly) says they are under water. Spreads of 120 and above,
  steep waves, 32 waves, 1.2 m wavelengths and 1.8 m amplitudes all render and label
  correctly. `factory/seastate.py` enforces the minimum. `wf_probes.py` spawns striped
  1 m poles around the whale to read the rendered water level off an image when in doubt.

## Water Realism

What the camera sees of open ocean is, in order of how much it matters for the whale/
background boundary the model has to learn:

1. **Colour and depth.** Open ocean is dark blue-grey, not turquoise: light is absorbed
   within metres (red first) and little is scattered back. `MI_OceanDeep` sets the
   plugin's `Absorption`, `Scattering` and `Water Albedo` for that. Randomise these per
   shot (coastal green, deep blue, overcast grey) via the shot JSON `ocean` key.
2. **Surface shape.** A Gerstner spectrum with long swell plus short wind chop
   (`OCEAN_WAVES_PARAMS`). The label pass evaluates the same waves, so changing the
   spectrum is free for labelling. Randomise amplitude/wavelength/wind angle per shot.
3. **Sun glitter and sky reflection.** Mostly the normal-detail strength and roughness
   parameters (`Default Near/Distant Normal Strength`, `Water Roughness`) and sun
   elevation. Low sun plus the camera looking toward it gives the glitter field seen in
   many field photos.
4. **Whitecaps and foam.** The plugin's `Enable Ocean Foam` switch puts texture foam on
   crests, but close to the camera it renders as beige blocks, so it is off; whitecaps
   are an open item (a custom foam texture and colour on `MI_OceanDeep`). The whale's own
   white water is calculated plume geometry (`factory/splash.py`); a Niagara fluid would
   look better still, and the coverage-twin trick for labels carries over (any spray
   material can write custom stencil 2).
5. **Camera and lens.** Telephoto focal lengths, shallow depth of field, motion blur,
   exposure jitter, slight vignetting and sensor noise; these come from Movie Render
   Queue settings and post-process, and matter more for sim-to-real than extra water
   detail. Render with 4-8 spatial samples for final datasets (anti-aliased edges).

If the Water plugin tops out: paid Fab ocean systems (Oceanology, Fluid Flux) give FFT
spectra, foam and spray, but their surfaces are not the plugin's Gerstner waves, so the
analytic water test in the label shader would have to be replaced (e.g. by rendering the
water's depth into a dedicated pass). An alternative that costs nothing in Unreal is
compositing: render the whale with its waterline mask over *real* ocean photographs for
part of the dataset; the background distribution is then exactly real, at the price of a
fake whale/water contact line.

## Proposed Repository Structure

```text
Whale-Image-Data-Factory/
  PLAN.md
  README.md
  unreal/
    WhaleFactory/                  # UE project (.uproject, Config/, Content/)
      Content/WhaleFactory/        # maps, materials (incl. label PP material), sequences
      Content/Python/              # editor-side scripts that run inside UE
        render_shotlist.py         # read shot list -> set scene -> queue MRQ renders
  factory/                         # pure-Python package, runs outside UE, unit-testable
    config/
      poc_boat.yaml                # scenario config: ranges for camera/env/whale params
    shotlist.py                    # seeded sampling of shots -> shotlist.jsonl
    cameras.py                     # viewpoint samplers: boat (POC), drone (later)
    environment.py                 # sun/sky/sea-state/water-color randomization params
    postprocess.py                 # raw passes -> label PNGs, validation, per-shot stats
    dataset.py                     # manifest, near-duplicate-aware splits, export
    qa.py                          # overlays, contact sheets, distribution plots
  training/
    train_seg.py                   # DINOv3 segmentation fine-tune
    eval_real.py                   # metrics on real labeled photos vs classical baseline
  tests/
  data/                            # gitignored: renders, exports, checkpoints
```

## Dataset Output Format

```text
data/renders/<run_id>/
  run_config.yaml          # frozen copy of the config + seed + UE/asset versions
  shotlist.jsonl           # every sampled shot, including ones later filtered out
  images/000123.png        # RGB, sRGB, tonemapped
  masks/000123.png         # uint8: 0 = background, 1 = whale, 255 = ignore
  amodal/000123.png        # uint8: full whale silhouette (not for training in POC)
  meta/000123.json         # camera pose/intrinsics, whale pose/time, env params,
                           # visible/amodal pixel counts, visible fraction, bbox
  manifest.jsonl
```

Class IDs stay stable once defined. Future classes (boat, island, other whale instances)
get new IDs instead of renumbering.

## Shot Sampling (POC: Boat Viewpoint)

All sampling happens outside Unreal from a seed, so every dataset can be regenerated exactly.

| Parameter | POC range (to be tuned against real photos) |
| --- | --- |
| Distance to whale | 15–150 m |
| Bearing around whale | 0–360° |
| Camera height above water | 1.5–4 m (boat deck plus a standing photographer) |
| Aim | at a point on the whale body, with jitter so it isn't always centered |
| Roll | ±5° |
| Focal length | about 70–400 mm full-frame equivalent (telephoto, as in field photos) |
| Whale animation time | uniform over the surfacing cycle, weighted toward "partly above water" |
| Sun elevation / azimuth | 5–75° / 0–360° |
| Cloud cover, haze | clear to overcast |
| Sea state | `glassy`, `calm`, `light`, `moderate`, `rough` presets (`factory/seastate.py`), weighted toward light and moderate |
| Water color / scattering | randomized within plausible ocean ranges |
| Exposure | ±1 EV jitter |

Mix in about 10–15% negatives (whale fully submerged or out of frame) so the model learns
that "no whale" is a valid answer.

Low camera heights mean grazing view angles, so waves hide parts of the whale often.
These shots carry most of the interesting occlusion labels.

## Whale Motion (POC)

Use a skeletal whale mesh with a looping swim-cycle animation, moved along a path in
Sequencer. A scripted depth profile repeats a realistic cycle: travel just below the
surface, rise, roll the back and dorsal fin above the surface, then dive. The fluke-up dive
and breaching come later and depend on available animations. Each shot samples a
time `t` in this sequence, which sets the whale's pose and depth.

## Rendering

- Movie Render Queue, legacy settings ("PRESET") config asset `MRQ_RgbAndLabel`, rendered
  from the command line in game mode (`tools/render_sequence.sh`):
  `UnrealEditor <uproject> <Map> -game -LevelSequence=... -MoviePipelineConfig=... -RenderOffscreen`.
  It takes about 13 s per short sequence, including startup. Gotchas found:
  - `SceneCapture2D` does not render the Water plugin's ocean, so it is useless for RGB.
  - MRQ inside the headless editor (PIE executor) renders nothing.
  - A 1-frame sequence renders nothing, because the frame is used to prime motion blur.
    Use 3 or more frames.
  - Post-process passes that replace the tonemapper (`BL_REPLACING_TONEMAPPER`) never
    complete in MRQ. Use `BL_SCENE_COLOR_AFTER_TONEMAPPING`.
  - `{sequence_name}` resolves to "" in command-line renders, so files are named
    `<pass>.<frame>.png`.
- Shots with a camera cut each need warm-up frames (64 now) so Lumen and exposure settle.
- Passes per shot: final RGB and the visible-whale label pass. The amodal pass is to do.
- Target for the POC: about 2,000 shots at 1024–1536 px on the long side. Training crops
  and resizes from there.

## Disk Safety

`/home` is about 87% full and is **shared with other users on this machine**, so free
space can shrink even when we are idle. Anything that can write many GiB goes through
`factory/diskguard.py`:

| Threshold (free space on `/home`) | Default | Effect |
| --- | --- | --- |
| `WF_MIN_FREE_GIB` | 30 GiB | tools refuse to start a job that would leave less |
| `WF_WARN_FREE_GIB` | 40 GiB | guarded processes log warnings |
| `WF_KILL_FREE_GIB` | 15 GiB | guarded processes (editor, renders, unzip) are terminated |

- **Engine install:** the engine is installed on the root disk at
  `/var/tmp/<user>-unreal/UE_<version>` (about 298 GiB free there). `/var/tmp` has no
  age-based cleanup on this machine; only `/tmp` has the 30-day rule. The install
  directory is mode 700. The zip downloads to `~/Downloads`, because snap Firefox can only
  write under home. `install_unreal.sh` checks the zip's exact uncompressed size before
  extracting. Delete the zip afterwards.
- **Watchdog scope:** guarded processes watch both `/home` and the engine's filesystem.
- **Caches:** the editor's local derived-data cache is pinned to
  `unreal/WhaleFactory/DerivedDataCache/` (gitignored, safe to delete). Shader debug dumps
  are disabled. Core dumps are disabled for UE processes.
  UE 5.8 actually caches through a local **Zen** server. By default its data goes to
  `~/.config/Epic/UnrealEngine/Common/Zen/Data` on /home, and it ignores the DDC path.
  `tools/ue_exec.sh` passes `-ZenDataPath=/var/tmp/<user>-unreal/ZenData` on every
  launch, which moves it to the root disk. That is verified from the zenserver
  `--data-dir`; the `UE-ZenDataPath` env var alone did not work. The zenserver binary
  itself (about 0.3 GiB) still installs under `~/.config/Epic`. Zen garbage-collects
  entries unused for 14 days.
  Launching the raw `UnrealEditor` binary skips all of this.
- **Renders:** the capture pipeline must call `diskguard.estimate_render_bytes` and
  `ensure_free` before a batch. The upper bound for the POC (2,000 shots at 1536×1024) is
  about 15 GiB.
- **Git:** a pre-commit hook blocks `data/`, third-party assets, and any non-LFS file over
  5 MiB. LFS keeps a second local copy of each asset in `.git/lfs` (`git lfs prune` trims it).

## Phases

### Phase 0 — Environment and De-risking Spikes

- [ ] `NOAA_whales/UnrealEngine` is a completed source clone of UE 5.8.3 (`release`
      branch, 41 GB on /home, mostly git history; `Setup.sh` not run). Kept per user.
      A source build would need about 100+ GB more, which /home doesn't have, so we use
      the precompiled 5.8.3 build instead. Decide later whether to delete it or move it
      to /var/tmp to free 41 GB on /home.
- [x] Installed precompiled UE 5.8.3 at `/var/tmp/sat3737-unreal/UE_5.8.3` (71.7 GiB).
      `tools/check_unreal.sh` passes headless: the project loads, all 5 plugins mount,
      Python runs, and `r.CustomDepth=3`. The 37 GiB zip is still in
      `/var/tmp/sat3737-unreal/` until the editor is confirmed working in FastX.
- [x] Editor runs in the FastX remote desktop session and renders (confirmed by user).
- [x] Headless `-RenderOffscreen` rendering works over SSH (Vulkan on the RTX 4000 Ada).
- [x] **Label spike (passed, with the stand-in whale instead of a cube):** the mask covers
      exactly the above-water part, with the waterline within 1–4 px of the RGB image. This
      needed the analytic wave test, not the planned SceneDepth test (see "Core Technique").
- [ ] Delete the 37 GiB engine zip in `/var/tmp/sat3737-unreal/` (the editor works).
- [x] Repo hygiene: UE `.gitignore` entries, Git LFS for `.uasset`/`.umap`/art (git-lfs
      3.8.0 installed in `~/.local/bin`), and a pre-commit guard. Licensed assets stay out
      of the public repo, in `Content/ThirdParty/` (gitignored).
- [x] Disk guards: see "Disk Safety" below.
- [x] Found where the derived-data cache goes: UE 5.8 uses a local Zen server, now
      redirected to the root disk (see "Disk Safety").

### Phase 1 — Scene

- [x] UE project `unreal/WhaleFactory`; the scene is built by script
      (`tools/run_editor_script.sh build_scene`) into `/Game/WhaleFactory/Maps/OceanPOC`.
- [x] Ocean: WaterZone (4 km) + WaterBodyOcean with the plugin's `GerstnerWaves_Ocean`.
      The ocean spline is a *coastline*: inside it is a dry island. The default 200 m
      square left the whale in a dry hole, so the ocean actor (and its island) sit 1.5 km
      away. A dark sea floor sits 60 m down.
- [x] Ocean look: `MI_OceanDeep`, a child of the plugin's `Water_Material_Ocean`, with
      open-ocean absorption/scattering/albedo (`OCEAN_LOOK` in `wf_build_scene.py`), and our
      own Gerstner spectrum (`OCEAN_WAVES_PARAMS`) in `WW_Ocean`, a project copy of the
      plugin's wave asset. The ocean must point at that asset's inner `GerstnerWaterWaves`
      object: a `GerstnerWaterWaves` instanced directly on the ocean actor renders a surface
      that disagrees with the label pass and leaves holes around the whale (seen
      2026-10-10; the plugin's `WaterWavesAssetReference` cannot be retargeted from Python).
      Any shot JSON can override the material parameters (`ocean` key). Still to tune
      against real photos; see "Water Realism".
- [x] Sky Atmosphere, Volumetric Clouds, a directional sun, a SkyLight, and height fog.
- [x] Stand-in whale: `factory/procedural_whale.py` (our own geometry, safe to commit) with
      a dark wet-skin material, stencil 1, back about 0.8 m above calm water.
- [x] Humpback asset: "High-Poly Humbpack Whale" by K9239 (Sketchfab, CC-BY-4.0, credit
      required; THIRD_PARTY.md). Static mesh, 16 m, no rig; 4K colour/roughness/normal.
      `factory/humpback_asset.py` converts it, `build_scene` imports it to
      `/Game/ThirdParty/Humpback` with a textured skin material (`RoughnessScale` hook).
- [ ] A rigged humpback (swim cycle, fluke curl) is still wanted for non-rigid poses;
      until then poses are rigid transforms (`factory/poses.py`).
- [x] Rigid surface poses (logging, blow, arch, fluke-up, breach, pec-up, spyhop, awash,
      submerged) with a framing boat camera, and sprite-based spray (blow spout, breach
      crown, fluke wash) that labels as background: `factory/poses.py` -> shot JSON ->
      `tools/render_shot.sh` (applies via `wf_apply_shot.py`, then renders).
- [ ] A Sequencer level sequence with the swim cycle and the surfacing depth profile
      (needs a rig).
- [x] Label post-process material plus custom stencil setup on the whale.
- [x] Boat preview camera + `LS_BoatPreview` sequence; `TopCamera` + `LS_TopDebug` for
      checking water coverage.

### Phase 2 — Capture Pipeline

- [x] `factory/shotlist.py` + `factory/config/poc_boat.json`: `shotlist.jsonl` plus render
      batches from a seed.
- [x] `wf_build_batch.py` + `tools/render_batch.sh`: a batch (one sea state and water
      look, 50 shots) becomes one level sequence with a camera cut per shot; MRQ renders
      it in one game session at about 2 s per shot.
- [x] `factory/postprocess.py`: label PNGs, amodal body-part masks, per-pixel canonical body
      coordinates and depth, keypoints with occlusion, orientation and the 6-DoF pose in the
      camera frame, all from projecting the mesh (`factory/project.py`, filmback calibrated),
      `meta/`, `manifest.jsonl`.
- [x] End-to-end smoke test (12 shots) passed 2026-10-11; `tools/build_dataset.sh` runs
      the whole thing. First full run: `data/renders/run01` (2000 shots, seed 1).

### Phase 3 — QA and Dataset Export

- [x] Overlay contact sheets (`python3 -m factory.qa sheet`).
- [x] Automatic checks per shot: visible ⊆ amodal (flag above 5 % outside), empty mask on
      a non-negative, mask touching the border; distributions in `summary.json`.
- [x] Splits grouped by sea state and water look (`factory/dataset.py`).
- [ ] Parameter distribution plots.

### Phase 4 — Real Evaluation Set (Essential)

- [ ] Hand-label about 50–100 real field photos from the Lightroom catalog under the *same*
      rule (above-surface whale only), using SAM-assisted labeling. Cover varied lighting,
      sea state, and distance.
- [ ] Freeze this set as the sim-to-real benchmark. Never train on it.

### Phase 5 — DINOv3 Fine-tune Baseline

- [ ] ViT-B/16 (weights already in `~/dinov3/weights`), frozen backbone with a linear head
      on patch features. Measure mIoU and whale IoU on synthetic val and the real eval set.
- [ ] Then try a light decoder (DPT-style) for sharper edges at 1/16 patch resolution,
      and optionally LoRA or partial unfreezing.
- [ ] Compare against the classical `whale_isolation.py` masks on the same real eval set.
- [ ] Failure analysis on real photos guides the next round of randomization.

### Phase 6 — Expansion (After POC)

- [ ] Boats and islands as occluders and distractors, with new class IDs and stencil values.
- [ ] Drone sampler: altitude about 15–60 m, near-nadir pitch. Revisit the label rule here,
      since drone work often wants submerged-but-visible whale pixels. The amodal pass
      already supports that.
- [ ] More species, behaviors (fluke-up, breach, blow), calves and pods, multiple whales.

## Toward Scale, Orientation and Body-Location Estimation

The segmentation dataset already carries what a second stage needs, so no re-render is
required when that work starts:

- `bodycoords/` gives every whale pixel its canonical body location (fraction along the
  body, angle around the axis), the dense-correspondence target that lets a feature
  (scar, barnacle cluster, pigment patch) found in one photo be compared only against the
  same body region in another. `amodal/` gives the coarse part label for free.
- `keypoints` plus `camera_intrinsics` support a PnP-style pose solve or a direct
  regression of `pose_in_camera`; `pixels_per_metre_at_whale` and `whale_length_m` are
  the scale targets (for real photos with unknown focal length, scale must come from
  known whale dimensions, so the length-normalised body fraction is the robust quantity).
- Still missing, in priority order: (1) a rigged model, since body flex (arching, fluke
  curl) changes where a body location appears and a rigid mesh cannot teach that;
  (2) texture variation (several individuals' pigmentation, synthetic scars and barnacle
  patches) so neither the segmenter nor the correspondence model keys on one skin;
  (3) real labelled photos for evaluation; (4) mild lens distortion and sensor noise
  augmentation at training time, which are cheaper applied there than rendered.

## POC Success Criteria

1. The label spike passes and masks are verified pixel-aligned with RGB.
2. Generate about 2,000 labeled shots from one config and seed, end to end, headless.
3. The DINOv3 model trained only on synthetic data beats `dark_anchor_local_growth_v4`
   in whale IoU on the real eval set. Stretch goal: close most of the remaining gap with a
   small real fine-tune.

## Open Decisions

- Decided: author in the editor on this server via FastX remote desktop; humpback; free
  asset first, paid if realism limits sim-to-real.
- Decided: the GitHub repo is public, so third-party assets are gitignored. Collaborators
  will need their own copy of the whale asset (document the source and version).
- Fallback engine if Unreal on Linux becomes a blocker: Blender plus BlenderProc
  (scriptable headless, native object-index masks, small install) is a viable plan B.
