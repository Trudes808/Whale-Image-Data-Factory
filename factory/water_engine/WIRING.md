# run06 sea in Unreal: the engine package (written, not run)

Everything here was prepared offline (no editor was started). The reference for every formula is
`factory/waterref.py` (`Renderer._shade`), which was tuned against real photographs; the HLSL files are line-by-line
ports of it. Numbers in brackets are the defaults in `waterref.Look`.

## 1. Assets to import (per sea-state bake, once; per shot: the contact layer)

`python3 -m factory.oceanbake bake <state> --seed S --out /var/tmp/sat3737-parts-cache/water_engine/bakes` writes:

| file | size | format | UE import settings |
|---|---|---|---|
| `T_Disp_c{0..5}.exr` | 512^2 | RGBA half, ZIP | Compression **HDR (RGBA16F, uncompressed)**, sRGB off, MipGen **SimpleAverage**, address **Wrap**, filter **Trilinear + aniso**, LOD group World (no streaming bias), Never Stream |
| `T_Mom_c{0..5}.exr` | 512^2 | RGBA half, ZIP | same (the moments must be box-filtered linearly: SimpleAverage, never Sharpen/Blur, never BC6H) |
| `T_Foam.exr` | 2048^2 | RGBA half, ZIP | same |
| `../common/T_Detail.png` | 1024^2 | RGBA8 | **TC_Masks** (linear, no sRGB), Wrap, SimpleAverage |
| `<shot>_contact.exr` (+ `.json`) | 576^2 over 72 m | RGBA half | HDR, sRGB off, **Clamp**, no mips needed (sampled at mip 0) |
| sky `*_sunless.png` (`engine_skies/`) | 2048 x 1024 | 16-bit RGB, radiance / 8 | HDR or 16-bit linear, sRGB off, Wrap U / Clamp V, full mips (the glossy blur uses them) |

A bake is ~30 MB on disk (12 x 1.4 MB moment/displacement EXRs + 13 MB foam), ~45 MB on the GPU with mips.
`layout.json` carries every scalar the material needs (below). Bakes per run: one per (sea state, seed); 5 states x
4 seeds = 20 bakes = 0.6 GB. Per-shot phase variety stays with the scene offset (`scene_offset_cm`), as today.

## 2. Material M_OceanFFT

Duplicate the plugin's `Water_Material_Ocean` (or make a new material with *Used with Water*); remove the Gerstner
WPO / normal graph; set the ocean water body's Gerstner wave asset to **zero waves**.

* **World Position Offset**: Custom node `OceanFFT_WPO.hlsl` (float3). Inputs: Absolute World Position (vertex, before
  offsets), `T_Disp_c0..c2`, `T_Contact`, `CascadeL` = layout `cascades_m[0:3]`, `TexN` = 512, `ContactOrigin`,
  `ContactSize`, `ContactOn`, `WorldScale` = 100. It returns cm.
* **Emissive**: Custom node `OceanFFT_Pixel.hlsl` (float3, additional outputs OutNormal, OutFoam, OutBodyT).
* **Specular 0, Base Color 0, Roughness 1** (see 4). Normal: OutNormal (only matters for whatever engine lighting is
  left on). Opacity / shading model: see 4.

Texture parameters: `T_Disp_c0..c5`, `T_Mom_c0..c5`, `T_Foam`, `T_Detail`, `T_Contact`, `T_Sky` (16 textures:
use **Shared: Wrap** sampler source on all wrap textures, Shared: Clamp on T_Contact, or the shader will exceed the
16-sampler limit; alternatively pack Disp/Mom into two Texture2DArrays of 6 slices).

Scalar / vector parameters (from `layout.json`, `sky_irradiance.json`, `waterref.Look`):

| parameter | value / source |
|---|---|
| CascadeL03, CascadeL45 | `cascades_m` = 1056.5, 211.3, 42.26, 8.452 / 1.6904, 0.33808 |
| TexN, GeoN | 512, 3 |
| WindDir | (cos, sin) of `spec.wind_dir_deg` |
| ResCov | residual covariance in world axes from `slopes.residual_upwind/crosswind` and the wind direction (0 when the bake's field already meets Cox-Munk, which the calibrated bakes do) x ResScale [1] |
| GustAmp, GustScale | [0.35], [70 m] |
| FoamTile, FoamAgedGain, DetailRepeat | `foam.tile_m` 211.3, 0.8, 1.5 |
| ContactOrigin, ContactSize, ContactOn | contact json `origin_m`, `size_m` (72), 1 |
| SunDir | unit vector towards the shot's sun (the existing sun actor's direction) |
| SunE, SkyE | `sky_irradiance.json` (normal sun irradiance / horizontal sky irradiance, in the sky texture's radiance units) |
| SkyYawRad | world azimuth of the HDRI's u = 0.5 for the shot (= the sky sphere's yaw) |
| SkyTexScale, SkyW | 8, 2048 |
| Rrs | water-leaving reflectance (sr^-1): 0.0014, 0.0030, 0.0140 (fitted to the real far-water hue 278 / chroma 25) |
| FoamAlbedo, SkyFoldDim | [0.8], [0.6] |
| ExtraTaps | 1 (= the hardware's 16x aniso, what the reference used); 2-4 splits the footprint along its major axis |
| TailW, TailS | slope-density mixture [0.05, 1.8] (Cox-Munk peakedness) |
| SteepW, SteepN | steep-facet glint population: 0 since round 2 (round 1: 0.12 made salt-like dots), 2 |
| ModStrength, SigmaLongUp | hydrodynamic modulation 0.8 (round 1: 1.2); sqrt(sum `slopes.per_cascade_upwind[1:4]`) |
| Subsamples, SkySat | 4 sub-footprints shaded separately along the footprint's long axis (glints on the actual crests, no filtered 'long exposure'); 1.4 saturation of the reflected sky |
| GlintArea, PixelAngle, Dist, SvPos, FrameSeed | discrete Poisson glints: GlintArea 0 since round 2 (round 1: 4e-7 m^2 gave uniform round discs); 1 / focal length in px, pixel distance (m), SV_Position, a per-shot seed |
| ParallaxC3, C3Sigma | 1 = parallax occlusion of cascade 3 (strongly recommended, see 5); std of cascade 3's Dz |

The sky sphere material (`M_SkyHDRI`) and this material must use the same radiance scale; the emissive is in the
sky texture's units x SkyTexScale x the sky intensity the scene uses for `M_SkyHDRI`.

## 3. Labels: the surface the render shows

The labels must evaluate exactly what the WPO displaces: cascades 0-2 at mip 0 + the contact height, at the vertex's
rest position. Python: `factory.oceanbake.ShotSurface` (`shot_surface(shot)` reads `shot["sea_bake"]` and
`shot["contact_layer"]`). Height agreement with the displaced mesh: < 1e-4 m (tests/test_oceanbake.py).

### Call sites that evaluate the Gerstner surface today (verified at HEAD 816b1ad; worktree line numbers in brackets
### as of 2026-11-17, the other agent is editing these files)

| site (HEAD) | what it does | replace with |
|---|---|---|
| factory/shotlist.py:447 (465) `waves = gerstner.generate_waves(shot["waves"])` | waves of the batch sea state | `surf = oceanbake.shot_surface(shot)` after setting `shot["sea_bake"]` (per batch: state + seed) |
| factory/shotlist.py:463-464 (481-482) | scene-offset search: `surface_height` along the body probe | `surf.height_cm(probe_x, probe_y)` (t is gone: the bake is frozen) |
| factory/shotlist.py:475 (494) | lift whale + camera by the height at the aim point | `surf.height_cm(aim_x, aim_y)` |
| factory/shotlist.py:487 (506), (520) | companion lift | same |
| factory/shotlist.py:491 (510), (524) | splash lift | same |
| factory/shotlist.py:494 (whitewater.spec call), 567 (603, 609) `whitewater.ensure_obj(..., gerstner.generate_waves(...))` | white-water carpet on the surface | pass `surf` instead of `waves` (see whitewater) |
| factory/whitewater.py:78, 97 | contact band `|z - surface_height| < 45 cm` | `surf.height_cm(x, y)` |
| factory/whitewater.py:149 `gerstner.surface_points(waves, gx, gy, t)` | carpet vertices on the wave surface | `surf.surface_points_cm(gx, gy)` |
| factory/postprocess.py:119-131 (265-277) `height_above_local_water` | whale point height above the local water (class 2) | `surf.height_cm(world_x, world_y)` |
| factory/postprocess.py:248-250 (420-436) | builds the waves for it | `oceanbake.shot_surface(meta)` |
| tools/water_check.py:27, 30 (`visible_by_raymarch`), 55 | label vs render QA | `surf.height_cm` in the ray-march |
| unreal/.../wf_build_batch.py:83-120 (98-135) `read_gerstner_waves`, `gerstner_displacement`, `surface_height` | editor-side lift (`ride_waves`, 122-140) | not needed when shotlist has lifted (it returns early on `water_surface_z_cm`); delete or make it read the bake via `factory.oceanbake` (pure numpy, runs in the editor's Python) |
| unreal/.../wf_build_batch.py:178 (193) `waves = read_gerstner_waves()` | | load the bake's layout instead |
| unreal/.../wf_shot.py:211 (225) `apply_waves` | sets the Gerstner generator | set zero waves + set the bake's texture / scalar parameters on the ocean MID (`apply_bake(batch["sea_bake"])`) |
| unreal/.../wf_build_scene.py:274-324 LABEL_SHADER ray-march | label pass: whale point under the analytic water? | `LabelVisibility_FFT.hlsl` (same structure; height from the baked textures + contact) |
| unreal/.../wf_dump_waves.py | debug dump of the plugin's waves | obsolete |

The `seastate.SEA_STATES` Gerstner presets stay for older runs; run06 shots carry `sea_bake` instead of `waves`.

## 4. The mirror-only reflection limit, and how this package avoids it

Lumen / SSR reflections on Single Layer Water are mirror reflections of the scene at the pixel's normal: a slope
distribution wider than the pixel is not integrated, so the sun shows up as a few tiny highlights or (with roughness)
a dim smear, never as a field of clipped glints; at grazing angles the far sea reflects one sky direction.
Here both are computed in the material from the slope moments, so they are right at every distance:
* glitter: Gaussian-mixture slope density (Cox-Munk peakedness) at the half vector, over the footprint's LEAN
  covariance + capillary residual + sun disc, Smith masking, exact Fresnel; optional discrete glints (Poisson count
  of sun images per pixel, `Look.glint_area_m2`) so near pixels sparkle and clip like a photograph;
* sky: 9-point Gauss-Hermite integration over the visible facets, sky lat-long sampled at a mip matched to the lobe.
The engine's own specular is switched off (Specular 0). What is lost: reflections of scene objects (the whale's own
reflection at the waterline, boats). If that matters, keep a small engine reflection weight (Specular 0.1-0.2) or
add the whale's reflection by a planar-reflection capture limited to the contact area.

## 5. Open questions only the engine can settle (most risky first)

1. **Does Single Layer Water with Specular 0 really remove the engine's reflection**, and does it accept an emissive
   that large (glints of 10^3 x the water)? UE's EnvBRDF keeps a grazing Fresnel term even at F0 = 0. Test: flat sea,
   emissive 0, look at the horizon. Fallback: Unlit opaque material (then the water body comes from Rrs in the
   emissive, and the submerged whale is no longer seen through the water: needs a separate pass or a translucent
   variant).
2. **Occlusion of the 0.28-1.4 m waves (cascade 3)**: the reference shows that the visible pattern at grazing view
   depends on these waves hiding their own backs (far.coherence 0.895 with cascades 0-2 as geometry, 0.864 with 0-3,
   0.856 with 0-4). The planned mesh cannot carry them at 50-400 m. `ParallaxC3` (32-step parallax occlusion in the
   pixel shader) is the proposed substitute; its cost and its look at 0.5-3 degrees grazing need an engine test. The
   labels ignore cascade 3 (heights of 1-5 cm: the whale's waterline moves by less than that).
3. **Water body colour**: if SLW is kept, its absorption / scattering must reproduce `Rrs` (we fit Rrs, not the SLW
   coefficients); otherwise set SLW scattering to 0 and let the emissive carry the body (OutBodyT is the weight).
4. **Mesh tessellation vs the label surface**: the WPO displaces vertices; the label evaluates the continuous bilinear
   surface. With the ocean quadtree at tessellation 8 / LOD scale 16 the vertex spacing near the whale must be
   <= 25 cm for the 1.4-7 m waves of cascade 2 to be within ~1 cm. Check `tools/water_check.py` IoU on a test batch.
5. **Sampler budget and anisotropy**: 16 textures in one material (shared samplers), r.MaxAnisotropy = 16 actually
   applied to these textures; ExtraTaps cost.
6. **Half-float precision of world UVs** far from the origin: UVs are q / L with q up to a few km; use LWC-safe
   `DFDemote` positions relative to a tile origin if the scene offset grows.
7. **Per-shot contact textures in one Movie Render Queue batch**: textures cannot be keyed on a sequence track. Pack a
   batch's contact layers into a Texture2DArray (50 x 576^2 RGBA16F = 130 MB) and key the slice index through a
   Material Parameter Collection scalar track (the MPC track is keyable).
8. **Sky azimuth convention**: SkyYawRad must match the way `M_SkyHDRI` maps the HDRI (UE is left-handed: check that
   the sun disc of the sky sphere and the glitter line up on one test frame).
9. **Exposure / tone**: the reference develops with metering + sRGB + S-curve + saturation 1.15 (a camera JPEG);
   UE's filmic tone map and the factory's `camera_fx` must land on the same statistics: re-measure the first test
   batch with `experiments/realism_gap` and adjust exposure bias / saturation, not the water.
10. **Fog**: the reference hazes the water with exp(-3.912 d / 25 km) towards the horizon sky; the engine's exponential
   height fog on the opaque water should be set to the same visibility (25 km).

## 6. Estimate of the engine work

* import + material graph + parameters from layout.json (scripted like `M_SkyHDRI`): 1 day
* parallax occlusion of cascade 3 (ParallaxC3) and its cost at 2560 x 1707: 0.5-1 day
* WPO + labels (shader swap, Python call sites, ShotSurface in shotlist / whitewater / postprocess / water_check): 1 day
* reflection-model question (1-2 above) and colour match to the reference statistics: 1-2 days
* contact layer atlas + MPC keying, QA with water_check and the realism score card: 1 day
Total about 5-6 days of engine work, plus a test batch of ~200 shots to re-measure.
