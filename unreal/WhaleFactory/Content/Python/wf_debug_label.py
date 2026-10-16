"""Diagnostic: rebuild the label material so that it outputs what the analytic water sees, render, inspect.
WF_DEBUG_LABEL selects the probe:
  numwaves   -> NumWaves of water body BODY_INDEX / 32            (0 = the body data is empty in the render)
  numwaves1  -> the same for water body index 1
  amplitude  -> summed amplitude of BODY_INDEX / 200 cm
  height     -> (analytic surface height at the pixel's whale point + 100) / 200, whale pixels only
Restore with tools/run_editor_script.sh rebuild_labels afterwards.
"""
import os

import unreal

os.environ["WF_IMPORT_ONLY"] = "1"
import wf_build_scene  # noqa: E402

PROBES = {
    "numwaves": "FWaterBodyData B = GetWaterBodyData(BODY_INDEX); return saturate(B.NumWaves / 32.0);",
    "numwaves1": "FWaterBodyData B = GetWaterBodyData(1); return saturate(B.NumWaves / 32.0);",
    "amplitude": "FWaterBodyData B = GetWaterBodyData(BODY_INDEX); float A = 0; for (int w = 0; w < B.NumWaves; w++) A += GetWaveDataNew(w, B).Amplitude; return saturate(A / 200.0);",
    "px": "if (abs(Stencil.r - WHALE_STENCIL) > 0.5) return 0; float3 Dir = normalize(-CamVec); float3 P = CamPos + Dir * (CDepth.r / max(dot(Dir, ResolvedView.ViewForward), 1e-4)); return saturate((P.x + 50000.0) / 100000.0);",
    "py": "if (abs(Stencil.r - WHALE_STENCIL) > 0.5) return 0; float3 Dir = normalize(-CamVec); float3 P = CamPos + Dir * (CDepth.r / max(dot(Dir, ResolvedView.ViewForward), 1e-4)); return saturate((P.y + 50000.0) / 100000.0);",
    "pz": "if (abs(Stencil.r - WHALE_STENCIL) > 0.5) return 0; float3 Dir = normalize(-CamVec); float3 P = CamPos + Dir * (CDepth.r / max(dot(Dir, ResolvedView.ViewForward), 1e-4)); return saturate((P.z + 5000.0) / 10000.0);",
    "camx": "return saturate((CamPos.x + 50000.0) / 100000.0);",
    "rvcamx": "return saturate((DFDemote(ResolvedView.WorldCameraOrigin).x + 50000.0) / 100000.0);",
    "rvcamx2": "return saturate((LWCToFloat(ResolvedView.WorldCameraOrigin).x + 50000.0) / 100000.0);",
    "height": """if (abs(Stencil.r - WHALE_STENCIL) > 0.5) return 0;
float3 Dir = normalize(-CamVec); float RayLength = CDepth.r / max(dot(Dir, ResolvedView.ViewForward), 1e-4);
float3 P = CamPos + Dir * RayLength; float2 Q = P.xy; WaveOutput Wave = (WaveOutput)0;
for (int k = 0; k < 12; k++) { Wave = GetAllGerstnerWavesNew(BODY_INDEX, WaterTime, DFPromote(Q)); Q = P.xy - Wave.WPO.xy; }
return saturate((BASE_Z + Wave.WPO.z + 100.0) / 200.0);""",
}


def main() -> None:
    probe = os.environ.get("WF_DEBUG_LABEL", "numwaves")
    wf_build_scene.make_label_material(shader=PROBES[probe])
    unreal.log(f"[wf_debug_label] label material now outputs probe {probe!r}")
    unreal.log("[wf_debug_label] OK")


try:
    main()
finally:
    if os.environ.get("WF_QUIT_WHEN_DONE"):
        unreal.SystemLibrary.quit_editor()
