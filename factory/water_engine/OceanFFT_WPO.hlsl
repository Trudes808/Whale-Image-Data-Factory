// OceanFFT_WPO.hlsl -- Custom node body for World Position Offset (vertex shader) of M_OceanFFT.
// Written for UE 5.8 (not compiled / not run yet). Output: float3, connect to World Position Offset.
//
// Inputs (Custom node input pins):
//   RestPos     float3  Absolute World Position (vertex), i.e. the water mesh vertex BEFORE any offset, in cm
//                       (use the LWC-demoted value: DFDemote / "Absolute World Position" node with "Excluding Material Offsets")
//   T_Disp_c0, T_Disp_c1, T_Disp_c2   Texture2D (texture objects) -- the geometric cascades of the bake
//   T_Contact   Texture2D  per-shot contact layer (R = height m), clamp addressing
//   CascadeL    float3  L of cascades 0..2 in metres (layout.json cascades_m[0:3])
//   TexN        float   512 (layout.json n)
//   ContactOrigin float2 metres (contact json origin_m), ContactSize float (m), ContactOn float (0/1)
//   WorldScale  float   100 (cm per m)
//
// Exactly what factory.oceanbake.BakedSea.displacement + factory.contact.height_at evaluate for the labels:
// bilinear, mip 0, half precision, texel (i, j) at world (i L / N, j L / N) m -> UV = q / L + 0.5 / N.
// The Water plugin's own Gerstner waves must be off (zero waves in the water body's wave asset).

float2 q = RestPos.xy / WorldScale;                    // metres
float3 D = 0;
D += Texture2DSampleLevel(T_Disp_c0, T_Disp_c0Sampler, q / CascadeL.x + 0.5 / TexN, 0).xyz;
D += Texture2DSampleLevel(T_Disp_c1, T_Disp_c1Sampler, q / CascadeL.y + 0.5 / TexN, 0).xyz;
D += Texture2DSampleLevel(T_Disp_c2, T_Disp_c2Sampler, q / CascadeL.z + 0.5 / TexN, 0).xyz;
if (ContactOn > 0.5)
{
    float2 cuv = (q - ContactOrigin) / ContactSize;
    if (all(cuv > 0.0) && all(cuv < 1.0))
        D.z += Texture2DSampleLevel(T_Contact, T_ContactSampler, cuv, 0).r;
}
return D * WorldScale;                                  // cm
