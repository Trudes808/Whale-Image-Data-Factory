"""Create the Movie Render Queue config asset used by tools/render_sequence.sh.

Run headless with ``tools/run_editor_script.sh make_render_config``. The config renders
the final RGB image plus the whale label pass (M_WhaleVisibleMask) from the same frame,
so the two are pixel-aligned, and writes PNGs named ``<pass>.<frame>.png`` into
Saved/MovieRenders/wf, from where tools/render_sequence.sh moves them.
"""

import os
import traceback

import unreal

CONFIG_DIR = "/Game/WhaleFactory/Render"
CONFIG_NAME = "MRQ_RgbAndLabel"
LABEL_MATERIAL = "/Game/WhaleFactory/Labels/M_WhaleVisibleMask"
WIDTH, HEIGHT = 1280, 720
# Frames rendered and discarded after each camera cut, so Lumen, exposure and water settle.
WARMUP_FRAMES = 64
BATCH_WARMUP_FRAMES = 32
OUTPUT_DIR = "{project_dir}/Saved/MovieRenders/wf"


def log(message: str) -> None:
    unreal.log(f"[wf_make_render_config] {message}")


def make_config(name: str, with_label: bool, label_material: str = LABEL_MATERIAL, warmup: int = WARMUP_FRAMES,
                resolution=(WIDTH, HEIGHT), spatial_samples: int = 1) -> None:
    path = f"{CONFIG_DIR}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    config = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, CONFIG_DIR, unreal.MoviePipelinePrimaryConfig, None)
    if config is None:
        raise RuntimeError("could not create the MRQ config asset")

    deferred = config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    if with_label:
        label = unreal.MoviePipelinePostProcessPass()
        label.set_editor_property("enabled", True)
        label.set_editor_property("material", unreal.EditorAssetLibrary.load_asset(label_material))
        deferred.set_editor_property("additional_post_process_materials", [label])

    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.set_editor_property("output_resolution", unreal.IntPoint(*resolution))
    # {sequence_name} resolves to "" in command-line renders, so name by pass and frame;
    # tools/render_sequence.sh moves each render's files to its own output folder.
    output.set_editor_property("file_name_format", "{render_pass}.{frame_number}")
    directory = unreal.DirectoryPath()
    directory.set_editor_property("path", OUTPUT_DIR)
    output.set_editor_property("output_directory", directory)
    output.set_editor_property("override_existing_output", True)

    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.set_editor_property("engine_warm_up_count", warmup)
    aa.set_editor_property("render_warm_up_count", warmup)
    aa.set_editor_property("render_warm_up_frames", True)
    aa.set_editor_property("spatial_sample_count", int(spatial_samples))   # > 1 supersamples each frame (run03: fine wave chop under long lenses aliased into streaks)
    aa.set_editor_property("temporal_sample_count", 1)

    settings = [s.get_class().get_name() for s in config.get_all_settings()]
    unreal.EditorAssetLibrary.save_asset(path, only_if_is_dirty=False)
    log(f"saved {path} with settings {settings}")


def main() -> None:
    make_config(CONFIG_NAME, with_label=True)
    # RGB only, for diagnosing the render path without the label pass.
    make_config("MRQ_RgbOnly", with_label=False)
    # Dataset batches (tools/render_batch.sh): many camera cuts per sequence, so a shorter
    # per-cut warm-up; MRQ warms up at every camera cut.
    make_config("MRQ_Batch", with_label=True, warmup=BATCH_WARMUP_FRAMES)
    # supersampled variants for the run03 anti-aliasing test (WF_RENDER_CONFIG=MRQ_Batch_AA4)
    for n in (2, 4):
        make_config(f"MRQ_Batch_AA{n}", with_label=True, warmup=BATCH_WARMUP_FRAMES, spatial_samples=n)
    log("OK")


try:
    main()
except Exception:
    unreal.log_error("[wf_make_render_config] " + traceback.format_exc())
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
