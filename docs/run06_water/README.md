# Run06 water: what web-ocean gives us, and how it would go into the factory

Assessment of 2026-10-21 (user question: can the web-ocean demo raise the fidelity of our water, can it be combined
with the splashes and poses for a run06, and can the whale itself displace and splash the surface?).

## What web-ocean is

`NOAA_whales/web-ocean` (MIT, Noniv/web-ocean) is a stand-alone WebGPU renderer, 2 400 lines of JavaScript / WGSL, no
engine. The parts that matter for us:

- **Waves**: a Tessendorf FFT sea, six 256 x 256 cascades with patches from 1 103 m down to 0.26 m, a fetch-limited
  JONSWAP wind spectrum with Donelan-Banner directional spreading plus a long-crested swell, capillary dispersion and
  choppy horizontal displacement. Thousands of wave components per cascade.
- **Whitecaps**: foam injected where the surface Jacobian folds (crests about to break), stored in world-space maps,
  decaying over ~5 s, drifting downwind, breaking into streaks, textured with a procedural bubble lace.
- **Shading**: slope-moment (LEAN) roughness that stays right at every distance, sun glitter from an anisotropic slope
  distribution, turbid water body lit by the sky, backlit-crest subsurface glow.
- **Sky**: Hillaire atmosphere and two cloud layers (we use real HDRIs instead; not needed).

What it does not do: no objects in the water, no interaction, no shoreline (the README says so). Nothing of it runs
inside Unreal; it is a reference implementation whose maths is small enough to port.

## Where our run05 water falls short, seen next to it

`run05_vs_webocean_water.png`: three run05 frames (calm, moderate, rough) beside web-ocean rendered with the same
camera height, field of view, sun elevation and azimuth, and a wind / swell matching the sea state.

- Our ocean is the Unreal Water plugin's Gerstner sum: 16 components per sea state. At long lenses (the 352 mm shot)
  that reads as parallel corduroy ridges; the FFT surface has the irregular, multi-scale, streaky texture of a real
  wind sea. `fft_vs_gerstner.png` shows the same difference as height fields on a 256 m patch.
- Our open sea has no whitecaps at all (the plugin's crest foam is switched off because it renders as blocks near the
  camera); the real photographs at moderate and rough states have them everywhere, and they are exactly the kind of
  white blobs a whale detector must learn to ignore. web-ocean's Jacobian foam gives 0.3 % coverage at 9 m/s and
  1.8 % at 13 m/s with its default constants, in the range of real whitecap coverage.
- The water-only web-ocean renders themselves are not photo-matched (exposure, haze and turbidity were not tuned), so
  judge the wave structure and the foam, not the colour.

## The port: factory/fftocean.py

`factory/fftocean.py` is the simulation half of web-ocean in numpy: the same spectrum (JONSWAP + Donelan-Banner +
swell, the capillary dispersion, the per-cascade bands and gains), h0 generation, time evolution, choppiness, the
displacement / slope / Jacobian fields per cascade, periodic sampling at world points with the horizontal displacement
inverted (as `factory.gerstner.surface_height` does today, so the labels can use it unchanged in spirit), and the
whitecap history (injection where J < 0.68, decay 0.22 1/s, downwind drift). Our sea states map to wind / fetch /
swell as:

| sea state | U10 m/s | fetch km | swell Hs m | Hs of the FFT sea | Gerstner preset Hs (4 sigma) |
|---|---|---|---|---|---|
| calm | 3.5 | 40 | 0.5 | 0.48 | 0.23 |
| light | 6 | 60 | 0.9 | 0.88 | - |
| moderate | 9 | 90 | 1.4 | 1.52 | 1.11 |
| rough | 13 | 150 | 2.2 | 2.72 | 2.54 |

A six-cascade field set at 512 x 512 takes ~2 s per time step on the CPU; a 20 s whitecap history ~20 s.

## How it would go into Unreal (run06)

The Water plugin only knows Gerstner waves (`GerstnerWaterWaves` is the single wave class in UE 5.8.3; anything else
needs a C++ module). The route that fits our frozen-phase, per-batch pipeline:

1. **Bake per batch**: for the batch's sea state, seed and wave time, write the cascades as world-aligned textures
   (RGBA16F: Dx, Dy, Dz, Jxy per cascade; a slope / Jacobian set; the foam maps at 1 km and 128 m), ~10 MB per batch,
   2.5 GB for a run. The phases vary per shot through the existing per-shot scene offsets, as today.
2. **Material**: duplicate the plugin's `Water_Material_Ocean` into the project (our `MI_OceanDeep` already instances
   it) and add, with the same `MaterialEditingLibrary` scripting that builds `M_SkyHDRI`, a custom HLSL node that
   samples the cascades at world xy and adds them to World Position Offset after the Gerstner term, perturbs the
   normal from the slope textures, and a whitecap layer from the foam maps modulated by the bubble texture (base
   colour and roughness). The Gerstner amplitudes go to ~0 (the FFT's 1.1 km cascade carries the swell). The water
   quadtree mesh is already run at tessellation 8 / LOD scale 16; the 1.4 m and 0.26 m cascades stay normal-only.
3. **Labels**: `wf_build_batch.ride_waves`, `factory.whitewater` and the postprocess water plane evaluate the surface
   through `fftocean.height` instead of `gerstner.surface_height`; everything downstream (masks, keypoint states,
   height above water) is unchanged.
4. **Splashes and poses** are untouched: the Niagara / VAT splashes and the foam carpet are placed on whatever surface
   `height` returns, exactly as now.

Effort: 1-2 days for the bake + material + batch plumbing with test renders, half a day for the labels, a day of QA
(`tools/water_check.py` style: label water line vs render). The GPU is busy with run05 training for about a day, so a
first run06 test batch is realistic after that; the material work can be prepared in the editor meanwhile.

## The whale moving the water: factory/iwave.py

Yes, at heightfield level, and it fits the same route. `factory/iwave.py` is a prototype of Tessendorf's iWave:
linear deep-water waves (the vertical-derivative operator applied spectrally) with the whale's waterline
cross-section as a moving, pinned obstruction. Growing mask pushes water out (the body advancing or rising), a
shrinking one lets it fall back; foam is injected on steep slopes and along the moving waterline and decays.
`iwave_whale_interaction.png` runs it on the pose of run05 shot 002027:

- A: the whale surfacing at 2.2 m/s with 8 s of history gives a bow wave, flank wash and a V-shaped wake of ~10-50 cm
  amplitude, with a foam streak behind the body; shown composited on a calm FFT sea.
- B: a body slap / breach landing gives the ring wave and a foam disc.

For run06 this becomes a per-shot 64-80 m texture (height + foam, ~1 MB) centred on the whale, composited into the
same material as a fifth cascade and evaluated by the same `height` call for the labels; the whale's motion before
the shot (speed along its heading, rise rate, an impact for slaps and breaches) comes from the pose category, as the
whitewater carpet's wake length does today. Cost: ~20 s of CPU per shot at 512 x 512. The calibration (how much water
a given body motion pushes) is the open tuning; the prototype's constants are guesses.

What this is not: a free-surface fluid solver. Sheets, plunging crests, spray and droplets stay with the Niagara
splash meshes. A true 3D solver (Unreal's Niagara Fluids has FLIP and shallow-water templates; the Water plugin ships
a shallow-water "FluidSimulation" demo) would cost minutes per shot, would need a per-shot warm-up of the whale's
motion inside the editor, and would make the labels follow a simulation we do not control; feasible for a few hero
shots, not for a 10 000-shot run.

## Files

- `factory/fftocean.py`: the FFT sea (spectrum, evolution, sampling, whitecap history).
- `factory/iwave.py`: the interaction heightfield (solver, waterline mask from the posed body).
- `docs/run06_water/fft_vs_gerstner.png`: FFT height / whitecaps / Jacobian vs the Gerstner presets, three sea states.
- `docs/run06_water/run05_vs_webocean_water.png`: run05 frames beside matched water-only web-ocean renders.
- `docs/run06_water/iwave_whale_interaction.png`, `iwave_demo.npz`: the interaction prototype on a run05 pose.
- Headless web-ocean renders: Playwright's Chromium with `--enable-unsafe-webgpu --enable-features=Vulkan
  --use-angle=vulkan --ignore-gpu-blocklist` renders `ocean.html` here (`?ui=0&w=&h=&cam=&fov=&windSpeed=...`).
