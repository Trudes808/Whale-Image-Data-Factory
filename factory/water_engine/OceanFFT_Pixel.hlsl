// OceanFFT_Pixel.hlsl -- Custom node body for the pixel shading of M_OceanFFT (UE 5.8). NOT compiled or run yet.
// It is a line-by-line port of factory/waterref.py Renderer._shade (the reference the engine is matched to);
// when you change one, change the other.
//
// Output (return): float3 radiance in the sky texture's units x SkyTexScale (connect to Emissive Color).
// Additional outputs (Custom node "Additional Outputs"):
//   OutNormal  float3  tangent-space-free world normal of the filtered mesh + mean fine slope (for the engine's
//                      own lighting / SSR of nearby objects, if kept), OutFoam float (0..1 foam opacity),
//                      OutBodyT float (1 - effective Fresnel: weight of the engine's water body if it is used).
//
// Why emissive: Lumen / SSR reflections of Single Layer Water are mirror-like (roughness cannot widen sun
// glitter, and the sun's reflection is a single highlight). Here the sun glitter (Gaussian slope density at the
// half vector over the LEAN-filtered slope moments + capillary residual + sun disc) and the sky reflection
// (Gauss-Hermite 3x3 over the visible facets, sky lat-long at a blur matched to the lobe) are computed from the
// moment maps at every distance, so the engine's specular is set to 0 (see WIRING.md "reflection").
//
// Input pins:
//   Q            float3  Absolute World Position EXCLUDING material offsets (pixel), cm  -> rest position
//   Vw           float3  normalize(CameraPosition - AbsoluteWorldPosition(including offsets))  (= -CameraVector... use CameraVector node: it points from the pixel to the camera)
//   T_Disp_c0..c5, T_Mom_c0..c5  Texture2D (A of T_Disp = E[sx sy]; T_Mom = sx, sy, E[sx^2], E[sy^2])
//   T_Foam       Texture2D  211.3 m tile: R fresh, G aged
//   T_Detail     Texture2D  1.5 m repeat: R bubble density, B streak noise, A blend noise
//   T_Contact    Texture2D  per shot: R height, G foam, B damping, A wash (clamp)
//   T_Sky        Texture2D  the shot's HDRI lat-long WITHOUT the sun (radiance / SkyTexScale), full mip chain
//   CascadeL03   float4, CascadeL45 float2 (m), TexN float (512), GeoN int-as-float (3)
//   WindDir      float2 (unit, the direction the wind sea travels to)
//   ResCov       float3 (xx, yy, xy) capillary residual covariance in world axes (layout slopes.residual_*) x ResScale
//   GustAmp float (0.35), GustScale float (70 m), FoamTile float (211.3), FoamAgedGain float (0.8), DetailRepeat (1.5)
//   ContactOrigin float2 (m), ContactSize float (m), ContactOn float
//   SunDir float3 (unit, towards the sun), SunE float3 (normal irradiance, sky units), SkyE float3 (horizontal sky irradiance)
//   SkyYawRad float (world azimuth of the HDRI's u = 0.5), SkyTexScale float (8 = factory.skies.PNG_SCALE), SkyW float (texture width)
//   Rrs float3 (water-leaving reflectance, sr^-1: 0.0010, 0.0042, 0.0125), FoamAlbedo (0.8), SkyFoldDim (0.6)
//   ExtraTaps float (1..4): SampleGrad calls along the footprint's major axis (x the hardware's 16x aniso)
//   TailW (0.05), TailS (1.8): slope-density mixture (Cox-Munk peakedness); SteepW, SteepN: steep-facet glint population
//   ModStrength, SigmaLongUp: hydrodynamic modulation of cascades 4-5 by the up-wind slope of cascades 1-3
//                (SigmaLongUp = sqrt(sum layout slopes.per_cascade_upwind[1:4]))
//   GlintArea (m^2, 0 = off), PixelAngle (rad = 1 / focal length in px), Dist (pixel distance, m: length(CameraPos - WorldPos)/100),
//   SvPos (SV_Position.xy), FrameSeed: discrete glints (Poisson count of sun images per pixel)
//   ParallaxC3 (0/1), C3Sigma (m): parallax occlusion of cascade 3
//   Subsamples (4): sub-footprints shaded separately along the footprint's long axis; SkySat (1.4)

struct FOcean
{
    float2 dqdx; float2 dqdy; int extra;

    float4 SampleAniso(Texture2D T, SamplerState S, float2 uv, float L)
    {
        // hardware anisotropic sampling, optionally split into `extra` sub-footprints along the major axis
        float2 gx = dqdx / L, gy = dqdy / L;
        if (extra <= 1) return T.SampleGrad(S, uv, gx, gy);
        bool xmaj = dot(gx, gx) > dot(gy, gy);
        float2 maj = xmaj ? gx : gy;
        float2 gx2 = xmaj ? gx / extra : gx, gy2 = xmaj ? gy : gy / extra;
        float4 acc = 0;
        for (int k = 0; k < extra; k++)
            acc += T.SampleGrad(S, uv + maj * ((k + 0.5) / extra - 0.5), gx2, gy2);
        return acc / extra;
    }

    float Fresnel(float c)
    {
        const float n = 1.333;
        c = clamp(c, 1e-4, 1.0);
        float s2 = 1 - c * c, ct = sqrt(max(1 - s2 / (n * n), 0));
        float rs = (c - n * ct) / (c + n * ct), rp = (n * c - ct) / (n * c + ct);
        return 0.5 * (rs * rs + rp * rp);
    }

    float SmithLambda(float3 w, float3 nm, float3 cov)
    {
        float wn = clamp(dot(w, nm), -1, 1);
        float2 h = w.xy - wn * nm.xy; float hl = max(length(h), 1e-6); float2 a = h / hl;
        float sig2 = a.x * a.x * cov.x + a.y * a.y * cov.y + 2 * a.x * a.y * cov.z;
        float cotv = wn / sqrt(max(1 - wn * wn, 1e-8));
        float aa = max(cotv, 1e-3) / sqrt(2 * max(sig2, 1e-8));
        return aa >= 1.6 ? 0 : (1 - 1.259 * aa + 0.396 * aa * aa) / (3.535 * aa + 2.181 * aa * aa);
    }

    float3 Sky(Texture2D T, SamplerState S, float3 d, float yaw, float blurDeg, float W)
    {
        float az = atan2(d.y, d.x) - yaw;
        float2 uv = float2(frac(az / (2 * PI) + 0.5), saturate(0.5 - asin(clamp(d.z, -1, 1)) / PI));
        float lod = log2(max(radians(blurDeg) * W / (2 * PI), 1.0));
        return T.SampleLevel(S, uv, lod).rgb;
    }
};

FOcean O;
float2 q = Q.xy / 100.0;
// Optional parallax occlusion for cascade 3 (0.28-1.4 m waves, too short for the WPO mesh): march the view ray over
// cascade 3's height field and move the shading point to the first visible facet. The reference renderer shows this
// is what breaks the long horizontal streaks at grazing view (far.coherence 0.895 -> 0.864, see the report).
if (ParallaxC3 > 0.5)
{
    float3 Vt = normalize(Vw);
    float2 dir = -Vt.xy / max(Vt.z, 0.004);          // horizontal metres per metre of descent along the ray
    float hmax = 3.0 * C3Sigma;                       // C3Sigma: std of cascade 3's Dz (layout: sqrt of var), ~1-5 cm
    float2 qq = q - dir * hmax; float zr = hmax;      // start above the highest crest
    const int NS = 32; float dz = 2 * hmax / NS;
    [loop] for (int k = 0; k < NS; k++)
    {
        float hc = T_Disp_c3.SampleLevel(T_Disp_c3Sampler, qq / CascadeL03.w + 0.5 / TexN, 0).z;
        if (zr <= hc) break;
        qq += dir * dz; zr -= dz;
    }
    q = qq;
}
O.dqdx = ddx(q); O.dqdy = ddy(q); O.extra = (int)clamp(ExtraTaps, 1, 4);
float3 V = normalize(Vw);
float L[6] = { CascadeL03.x, CascadeL03.y, CascadeL03.z, CascadeL03.w, CascadeL45.x, CascadeL45.y };
float hN = 0.5 / TexN;

// wind gusts (cat's paws) on the fine cascades: variance-normalised gain from the blend noise at a large scale
float gn = O.SampleAniso(T_Detail, T_DetailSampler, q / GustScale + float2(17.0 / 1024, 41.0 / 1024), GustScale).a;
float g = (1 + GustAmp * (2 * gn - 1)) / sqrt(1 + GustAmp * GustAmp / 3);
float damp = 0, cfoam = 0;
float2 csl = 0;
if (ContactOn > 0.5)
{
    float2 cuv = (q - ContactOrigin) / ContactSize;
    if (all(cuv > 0) && all(cuv < 1))
    {
        float4 c = T_Contact.SampleLevel(T_ContactSampler, cuv, 0);
        float e = ContactSize / 576.0;          // contact texel (m): slopes by central differences
        float hxp = T_Contact.SampleLevel(T_ContactSampler, cuv + float2(e / ContactSize, 0), 0).r;
        float hxm = T_Contact.SampleLevel(T_ContactSampler, cuv - float2(e / ContactSize, 0), 0).r;
        float hyp = T_Contact.SampleLevel(T_ContactSampler, cuv + float2(0, e / ContactSize), 0).r;
        float hym = T_Contact.SampleLevel(T_ContactSampler, cuv - float2(0, e / ContactSize), 0).r;
        csl = float2(hxp - hxm, hyp - hym) / (2 * e);
        damp = saturate(c.b); cfoam = clamp(c.g, 0, 1.5);
    }
}

// --- K sub-footprints along the footprint's long axis (round 2): each is filtered over 1/K of the footprint and shaded
// separately, so waves longer than a sub-footprint are shaded as the facets they are (no "long exposure"); glints then
// sit on the actual sun-facing crests. Subsamples = 4 in the reference; 1 = the single-sample material.
float3 LwAcc = 0; float FeffAcc = 0; float2 muAcc = 0; float3 nMeshAcc = 0;
float2 q0 = q;
FOcean O0 = O;
int Ksub = (int)clamp(Subsamples, 1, 8);
bool xmajK = dot(O0.dqdx, O0.dqdx) > dot(O0.dqdy, O0.dqdy);
float2 majK = xmajK ? O0.dqdx : O0.dqdy;
[loop] for (int kk = 0; kk < Ksub; kk++)
{
float2 qk = q0 + majK * ((kk + 0.5) / Ksub - 0.5);
O = O0; if (xmajK) O.dqdx /= Ksub; else O.dqdy /= Ksub;
float2 q = qk;
    float2 mu = csl, muGeo = csl; float3 cov = 0;
    // --- per cascade: (sx, sy, E sx^2, E sy^2) from T_Mom, E sx sy from T_Disp.a, footprint-filtered by the sampler
    #define CASCADE(c, TM, TMS, TD, TDS)                                                                        \
    {                                                                                                           \
        float2 uv = q / L[c] + hN;                                                                              \
        float4 m = O.SampleAniso(TM, TMS, uv, L[c]);                                                            \
        float  mxy = O.SampleAniso(TD, TDS, uv, L[c]).a;                                                        \
        float2 s = m.xy; float3 v = float3(max(m.z - s.x * s.x, 0), max(m.w - s.y * s.y, 0), mxy - s.x * s.y);   \
        if (c >= 3)                                                                                             \
        {   /* texture bombing: a second sample through a mirror / turn of the field, variance-preserving blend */ \
            float2x2 Tm = (c == 3) ? float2x2(cos(2 * wa), sin(2 * wa), sin(2 * wa), -cos(2 * wa))              \
                        : (c == 4) ? float2x2(-1, 0, 0, -1) : float2x2(-cos(2 * wa), -sin(2 * wa), -sin(2 * wa), cos(2 * wa)); \
            float2 off = (c == 3) ? float2(0.371, 0.613) : (c == 4) ? float2(0.519, 0.173) : float2(0.283, 0.797); \
            float2 q2 = mul(Tm, q) + off * L[c];                                                                \
            FOcean O2 = O; O2.dqdx = mul(Tm, O.dqdx); O2.dqdy = mul(Tm, O.dqdy);                                \
            float4 m2 = O2.SampleAniso(TM, TMS, q2 / L[c] + hN, L[c]);                                          \
            float  m2xy = O2.SampleAniso(TD, TDS, q2 / L[c] + hN, L[c]).a;                                      \
            /* slopes of f(T q + o): T^T grad f (T symmetric here); second moments T M T^T */                   \
            float2 s2 = mul(transpose(Tm), m2.xy);                                                              \
            float a_ = Tm[0][0], b_ = Tm[0][1], c_ = Tm[1][0], d_ = Tm[1][1];                                   \
            float3 M2 = float3(a_ * a_ * m2.z + c_ * c_ * m2.w + 2 * a_ * c_ * m2xy,                            \
                               b_ * b_ * m2.z + d_ * d_ * m2.w + 2 * b_ * d_ * m2xy,                            \
                               a_ * b_ * m2.z + c_ * d_ * m2.w + (a_ * d_ + b_ * c_) * m2xy);                   \
            float3 v2 = float3(max(M2.x - s2.x * s2.x, 0), max(M2.y - s2.y * s2.y, 0), M2.z - s2.x * s2.y);     \
            float bw = O.SampleAniso(T_Detail, T_DetailSampler, q / (2.37 * L[c]) + float2(3.1, 7.3) * c / 1024.0, 2.37 * L[c]).a; \
            float wa_ = cos(0.5 * PI * bw), wb_ = sin(0.5 * PI * bw);                                           \
            s = wa_ * s + wb_ * s2; v = wa_ * wa_ * v + wb_ * wb_ * v2;                                         \
            if (c == 3) sLong += s;                 /* modulation uses the un-gained slope, as waterref */    \
            float gain = g * (1 - (c >= 4 ? 0.9 : 0.5) * damp) * (c >= 4 ? gm : 1.0);                           \
            s *= gain; v *= gain * gain;                                                                        \
        }                                                                                                       \
        mu += s; cov += v; if (c < 3) muGeo += s; if (c == 1 || c == 2) sLong += s;                             \
    }
    float wa = atan2(WindDir.y, WindDir.x);
    float2 sLong = 0; float gm = 1;
    CASCADE(0, T_Mom_c0, T_Mom_c0Sampler, T_Disp_c0, T_Disp_c0Sampler)
    CASCADE(1, T_Mom_c1, T_Mom_c1Sampler, T_Disp_c1, T_Disp_c1Sampler)
    CASCADE(2, T_Mom_c2, T_Mom_c2Sampler, T_Disp_c2, T_Disp_c2Sampler)
    CASCADE(3, T_Mom_c3, T_Mom_c3Sampler, T_Disp_c3, T_Disp_c3Sampler)
    // hydrodynamic modulation: short waves rougher on the down-wind faces of the longer waves; E[gm^2] = 1
    { float zl = clamp(dot(sLong, WindDir) / SigmaLongUp, -3, 3); gm = exp(-ModStrength * zl - ModStrength * ModStrength); }
    CASCADE(4, T_Mom_c4, T_Mom_c4Sampler, T_Disp_c4, T_Disp_c4Sampler)
    CASCADE(5, T_Mom_c5, T_Mom_c5Sampler, T_Disp_c5, T_Disp_c5Sampler)
    #undef CASCADE
    cov += ResCov * g * g * gm * gm * (1 - 0.9 * damp);
    
    float3 nMesh = normalize(float3(-muGeo, 1));
    float vnm = max(dot(V, nMesh), 1e-3);
    
    // --- sun glitter
    float3 Hv = normalize(V + SunDir);
    float hz = max(Hv.z, 1e-3);
    float2 z = -Hv.xy / hz - mu;
    float sunVar = (0.004654 / 2) * (0.004654 / 2) / 4 / (hz * hz * hz * hz);
    float pdf = 0;
    {
        float base = 1 / (1 - TailW + TailW * TailS * TailS);
        [unroll] for (int t = 0; t < 2; t++)
        {
            float sc = t == 0 ? base : base * TailS * TailS, wt = t == 0 ? 1 - TailW : TailW;
            float cxx = cov.x * sc + sunVar, cyy = cov.y * sc + sunVar, cxy = cov.z * sc;
            float det = max(cxx * cyy - cxy * cxy, 1e-12);
            float ee = (z.x * z.x * cyy + z.y * z.y * cxx - 2 * z.x * z.y * cxy) / det;
            pdf += wt * exp(-0.5 * ee) / (2 * PI * sqrt(det));
        }
    }
    float Dm = pdf / (hz * hz * hz * hz);
    float sw = min(SteepW * gm * gm * g * g, 0.5);      // steep facets sit on the rough (modulated, gusty) patches
    Dm = (1 - sw) * Dm + sw * (SteepN + 2) / (2 * PI) * pow(hz, SteepN);
    float G = 1 / (1 + O.SmithLambda(V, nMesh, cov) + O.SmithLambda(SunDir, nMesh, cov));
    float glint = O.Fresnel(dot(V, Hv)) * Dm * G / (4 * vnm) * (dot(SunDir, nMesh) > 0 ? 1 : 0);
    float3 Lglint = SunE * glint;
    if (GlintArea > 0)
    {
        // discrete glints: N ~ Poisson(lambda) sun images of a facet patch GlintArea in the pixel's cross-section
        float sunE1 = dot(SunE, 1.0 / 3);
        float Lsun = sunE1 / (PI * 0.004654 * 0.004654);
        float Apix = (Dist * PixelAngle) * (Dist * PixelAngle);
        float one = Lsun * O.Fresnel(dot(V, Hv)) * min(GlintArea * max(dot(V, Hv), 0.05) / Apix, 1.0);
        float lam = sunE1 * glint / max(one, 1e-12);
        if (lam < 50)
        {
            // hash -> uniform, inverse-CDF Poisson (exact for the small lambdas where it matters)
            uint hsh = asuint(SvPos.x) * 1973u ^ asuint(SvPos.y) * 9277u ^ (uint)FrameSeed * 26699u;
            hsh ^= hsh >> 15; hsh *= 0x2c1b3c6du; hsh ^= hsh >> 12; hsh *= 0x297a2d39u; hsh ^= hsh >> 15;
            float u = (hsh & 0xffffff) / 16777216.0;
            float p = exp(-lam), cdf = p; int n = 0;
            [loop] for (; n < 64 && u > cdf; n++) { p *= lam / (n + 1); cdf += p; }
            Lglint *= n * one / max(sunE1 * glint, 1e-20);
        }
    }
    
    // --- sky reflection: Gauss-Hermite 3x3 over the slope distribution, weighted by visible projected area
    const float ghx[3] = { -1.7320508, 0, 1.7320508 };
    const float ghw[3] = { 1.0 / 6, 2.0 / 3, 1.0 / 6 };
    float a11 = sqrt(max(cov.x, 1e-10)), a21 = cov.z / a11, a22 = sqrt(max(cov.y - a21 * a21, 1e-10));
    float3 Lr = 0; float Fs = 0, Ws = 0;
    // keep the quadrature on visible facets: shift the lobe's centre towards the camera when the mean facet faces
    // away by more than one sigma (backs of crests at grazing view)
    float2 muR = mu;
    {
        float vxy2 = max(dot(V.xy, V.xy), 1e-8); float2 u = V.xy / sqrt(vxy2);
        float sigv = sqrt(max(u.x * u.x * cov.x + u.y * u.y * cov.y + 2 * u.x * u.y * cov.z, 1e-8));
        float ex = max(dot(mu, V.xy) - (V.z + sigv * sqrt(vxy2)), 0) / vxy2;
        muR = mu - ex * V.xy;
    }
    // one sky lookup at the Fresnel-weighted mean reflection of the 9 nodes, blurred to their angular spread (round 2)
    float3 Rm = 0; float R2 = 0, fb = 0;
    [unroll] for (int i = 0; i < 3; i++)
    [unroll] for (int j = 0; j < 3; j++)
    {
        float2 zi = muR + float2(a11 * ghx[i], a21 * ghx[i] + a22 * ghx[j]);
        float3 n = normalize(float3(-zi, 1));
        float vdn = dot(V, n);
        float w = ghw[i] * ghw[j] * max(vdn, 0) / n.z;
        float3 R = 2 * vdn * n - V;
        float below = R.z < 0 ? 1 : 0;
        R.z = R.z < 0 ? -R.z * 0.3 + 0.002 : R.z;
        float wf = w * O.Fresnel(max(vdn, 1e-4));
        Rm += R * wf; R2 += wf * dot(R, R); fb += wf * below; Fs += wf; Ws += w;
    }
    float FsC = max(Fs, 1e-12);
    float3 Rd = Rm / FsC;
    float spread = sqrt(max(R2 / FsC - dot(Rd, Rd), 0));
    float3 Lsk = O.Sky(T_Sky, T_SkySampler, normalize(Rd), SkyYawRad, min(degrees(spread) * 0.8, 22), SkyW) * SkyTexScale;
    float lumk = dot(Lsk, float3(0.2126, 0.7152, 0.0722));
    Lsk = max(lumk + (Lsk - lumk) * SkySat, 0) * (1 - (1 - SkyFoldDim) * fb / FsC);   // SkySat 1.4: the HDRI skies are paler than the tropical sky
    Ws = max(Ws, 1e-9); float Feff = Fs / Ws;
    Lr = Lsk * Feff;
    
    // --- water body (water-leaving reflectance x downwelling irradiance, through the surface)
    float3 Ed = SunE * max(SunDir.z, 0) + SkyE;
    float3 Lbody = Rrs * Ed * (1 - Feff) / 0.98;
    float3 Lw = Lr + Lbody + Lglint;
    
LwAcc += Lw / Ksub; FeffAcc += Feff / Ksub; muAcc += mu / Ksub; nMeshAcc += nMesh / Ksub;
}
O = O0;
float3 Lw = LwAcc; float Feff = FeffAcc; float2 mu = muAcc; float3 nMesh = normalize(nMeshAcc);

// --- foam: whitecaps (fresh dense, aged streaky lace) + the contact layer's foam
float lace = O.SampleAniso(T_Detail, T_DetailSampler, q / DetailRepeat, DetailRepeat).r;
float2 fs = O.SampleAniso(T_Foam, T_FoamSampler, q / FoamTile + 0.5 / 2048.0, FoamTile).rg;
float2 wd = WindDir; float2 sp = float2(dot(q, wd), dot(q, float2(-wd.y, wd.x)));
float streak = O.SampleAniso(T_Detail, T_DetailSampler, sp * float2(1.0 / 6.0, 1.0) / 9.0, 9.0).b;   // stretched 6:1 down-wind
float fresh = fs.x, aged = fs.y * (0.4 + 1.2 * streak);
float cf = saturate(fresh), ca = saturate(FoamAgedGain * aged);
float4 lc = O.SampleAniso(T_Detail, T_DetailSampler, q / 6.0 + float2(11.0, 5.0) / 1024, 6.0);      // 6 m lace: visible cells
float lace6 = saturate((lc.r - 0.25) / 0.5);
float alpha = cf * (0.45 + 0.55 * lace6) * (0.85 + 0.15 * lace) + (1 - cf) * ca * lace * (0.3 + 0.7 * lace6);
float foamTex = 0.72 + 0.4 * lc.g;
// contact foam: the amount sets a threshold on a two-scale bubble pattern -> broken patches with lacy edges
float4 lp = O.SampleAniso(T_Detail, T_DetailSampler, q / 2.0 + float2(23.0, 3.0) / 1024, 2.0);
float pat = 0.55 * lp.r + 0.25 * lace + 0.2 * lp.b;
float edgeW = 0.06 + 0.5 * min(2 * length(xmajK ? O0.dqdx : O0.dqdy), 0.3);
float ac = saturate((pat - (1 - min(cfoam * 1.4, 1.2)) + edgeW) / (2 * edgeW));
if (ac > alpha) foamTex = 0.65 + 0.45 * lp.g;
alpha = 1 - (1 - alpha) * (1 - ac);
float3 Lfoam = FoamAlbedo / PI * (SunE * max(dot(SunDir, nMesh), 0) * 0.6 + SkyE) * foamTex;
float3 Lout = lerp(Lw, Lfoam, alpha);

OutNormal = normalize(float3(-mu, 1));
OutFoam = alpha;
OutBodyT = 1 - Feff;
return Lout;     // aerial haze: the engine's exponential height fog on this opaque surface (see WIRING.md)
