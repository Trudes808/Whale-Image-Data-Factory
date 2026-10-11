# Third-party assets

Asset files are not committed (see `.gitignore`: `third_party_assets/` and
`unreal/WhaleFactory/Content/ThirdParty/`). Collaborators download them to the paths
below; the tools then import them into Unreal.

## Humpback whale model

This work is based on "High-Poly Humbpack Whale"
(https://sketchfab.com/3d-models/high-poly-humbpack-whale-5a6a09f1c27e4d0f890f2a30c10d7a81)
by K9239 (https://sketchfab.com/K9239) licensed under CC-BY-4.0
(http://creativecommons.org/licenses/by/4.0/).

Credit the author as above wherever renders or datasets made from it are shared.

| | |
| --- | --- |
| Download | Sketchfab, glTF format (`high-poly_humbpack_whale.zip`, 22 MB) |
| Put the zip at | `third_party_assets/humpback_k9239/high-poly_humbpack_whale.zip` |
| Extract to | `third_party_assets/humpback_k9239/src/` (`scene.gltf`, `scene.bin`, `textures/`) |
| Convert | `python3 -m factory.humpback_asset third_party_assets/humpback_k9239/src third_party_assets/humpback_k9239/converted` |
| Unreal assets | `/Game/ThirdParty/Humpback/` (made by `tools/run_editor_script.sh build_scene`) |

What the model is: a static mesh (no skeleton, no animation) of an adult humpback about
16.1 m long in a cruising pose with the pectoral fins hanging down, with 22 barnacle
meshes and two eyes, and 4K base colour, roughness and normal textures. About 52k
triangles. `factory.humpback_asset` bakes the glTF node transforms, converts to the
project's Unreal frame (+X nose, +Y right, +Z up, cm) and flips the normal map's green
channel for Unreal.
