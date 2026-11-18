// LabelVisibility_FFT.hlsl -- replacement for the Gerstner ray-march in wf_build_scene.LABEL_SHADER
// (unreal/WhaleFactory/Content/Python/wf_build_scene.py lines 274-324 at HEAD 816b1ad: the block from
// "// The water surface is not in SceneDepth here, ..." to the final "return Who;").
// NOT compiled or run. Same inputs as today (Stencil, CDepth, SDepth, CamVec, CamPos, BASE_Z, ANALYTIC_WAVES)
// plus the baked textures as texture-object pins:
//   T_Disp_c0, T_Disp_c1, T_Disp_c2, T_Contact; CascadeL (float3, m), TexN (512), ContactOrigin (float2 m),
//   ContactSize (m), ContactOn (0/1), MaxHeightCm (layout: sum over geo cascades of max |Dz| + contact max, cm).
// The surface is evaluated exactly as factory.oceanbake.ShotSurface.height_cm does (and as the WPO displaces the
// mesh vertices): horizontal displacement inverted by fixed-point iteration at mip 0, half-precision texels.

// ---- (unchanged head of LABEL_SHADER: Who / CDepth tests / #if !ANALYTIC_WAVES ... #endif) ----

struct FSea
{
    float3 Disp(Texture2D T0, SamplerState S0, Texture2D T1, SamplerState S1, Texture2D T2, SamplerState S2, float2 q, float3 Lc, float N)
    {
        float h = 0.5 / N;
        return T0.SampleLevel(S0, q / Lc.x + h, 0).xyz + T1.SampleLevel(S1, q / Lc.y + h, 0).xyz + T2.SampleLevel(S2, q / Lc.z + h, 0).xyz;
    }
};
FSea Sea;
float3 Dir = normalize(-CamVec);
float RayLength = CDepth.r / max(dot(Dir, ResolvedView.ViewForward), 1e-4);
float Ceiling = BASE_Z + MaxHeightCm + 1.0;
const int Steps = 128;
float3 P = CamPos + Dir * RayLength;
float fBand = 1.0;
if (CamPos.z > Ceiling && CamPos.z > P.z)
    fBand = saturate((Ceiling - P.z) / (CamPos.z - P.z));
float tStart = RayLength * (1.0 - fBand);
for (int i = Steps; i >= 0; i--)
{
    float3 S = CamPos + Dir * (tStart + (RayLength - tStart) * i / Steps);
    if (S.z > Ceiling) break;
    float2 x = S.xy / 100.0;              // m
    float2 q = x;
    float3 D = 0;
    for (int k = 0; k < 8; k++)           // q + D(q) = x  (the geometric cascades' horizontal Jacobian is << 1)
    {
        D = Sea.Disp(T_Disp_c0, T_Disp_c0Sampler, T_Disp_c1, T_Disp_c1Sampler, T_Disp_c2, T_Disp_c2Sampler, q, CascadeL, TexN);
        q = x - D.xy;
    }
    D = Sea.Disp(T_Disp_c0, T_Disp_c0Sampler, T_Disp_c1, T_Disp_c1Sampler, T_Disp_c2, T_Disp_c2Sampler, q, CascadeL, TexN);
    float hz = D.z;
    if (ContactOn > 0.5)
    {
        float2 cuv = (q - ContactOrigin) / ContactSize;
        if (all(cuv > 0) && all(cuv < 1)) hz += T_Contact.SampleLevel(T_ContactSampler, cuv, 0).r;
    }
    if (S.z < BASE_Z + 100.0 * hz - 0.1) return 0;
}
return Who;
