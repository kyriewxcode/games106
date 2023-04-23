#include "meshInput.hlsl"

float3 calculateNormal(VSOutput input)
{
    float3 tangentNormal = textureNormalMap.Sample(samplerNormalMap, input.UV).xyz * 2.0 - 1.0;

    float3 N = normalize(input.Normal);
    float3 T = normalize(input.Tangent);
    float3 B = normalize(cross(N, T));
    float3x3 TBN = transpose(float3x3(T, B, N));

    return normalize(mul(TBN, tangentNormal));
}

// Walter et al. 2007, "Microfacet Models for Refraction through Rough Surfaces"
float D_GGX(float roughness, float NoH)
{
    float oneMinusNoHSquared = 1.0 - NoH * NoH;
    float a = NoH * roughness;
    float k = roughness / (oneMinusNoHSquared + a * a);
    float d = k * k * (1.0 / PI);
    return d;
}

// Smith-GGX: [Smith 1967, "Geometrical shadowing of a random rough surface"]
float G_Smith_GGX(float roughness, float NoV, float NoL )
{
    float a2 = roughness * roughness;
    float G_SmithV = NoV + sqrt( NoV * (NoV - NoV * a2) + a2 );
    float G_SmithL = NoL + sqrt( NoL * (NoL - NoL * a2) + a2 );
    return 0.5 / (G_SmithV + G_SmithL);
}

// Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
float3 F_Schlick(const float3 f0, float VoH) {
    return f0 + (1.0f - f0) * pow(1.0 - VoH, 5.0f);
}

float3 Tonemap_ACES(const float3 c) {
    // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
    // const float a = 2.51;
    // const float b = 0.03;
    // const float c = 2.43;
    // const float d = 0.59;
    // const float e = 0.14;
    // return saturate((x*(a*x+b))/(x*(c*x+d)+e));

    //ACES RRT/ODT curve fit courtesy of Stephen Hill
	float3 a = c * (c + 0.0245786) - 0.000090537;
	float3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
	return a / b;
}
float4 main(VSOutput input) : SV_TARGET
{
    float3 albedo = textureColorMap.Sample(samplerColorMap, input.UV).rgb * material.baseColorFactor;
    float3 ambientAO = textureOcclusion.Sample(samplerOcclusion, input.UV).rrr;
    float3 emmisive = textureEmissive.Sample(samplerEmissive, input.UV).rgb * material.emissiveFactor;
    float2 metallicRoughness = textureMetallicRoughness.Sample(samplerMetallicRoughness, input.UV).rg;

    float roughness = metallicRoughness.g * material.roughnessFactor;
    float metallic = metallicRoughness.r * material.metallicFactor;
    float3 light = material.lightColor * material.lightIntensity;
    float3 f0 = 0.04 * (1.0 - metallic) + albedo.rgb * metallic;

    float3 n = calculateNormal(input);
    float3 l = normalize(ubo.lightPos.xyz - input.WorldPos);
    float3 v = normalize(ubo.viewPos.xyz - input.WorldPos);
    float3 h = normalize(v + l);
    float NoV = abs(dot(n, v)) + 1e-5;
    float NoL = saturate(dot(n, l));
    float NoH = saturate(dot(n, h));
    float LoH = saturate(dot(l, h));

    float D = D_GGX(roughness, NoH);
    float G = G_Smith_GGX(roughness, NoV, NoL);
    float3 F = F_Schlick(f0, LoH);
    float3 specular = D * G * F * light;

    float3 diffuse = albedo * ambientAO * saturate(dot(n, l)) * light;
    
    float3 color = emmisive + diffuse + specular;

    color = Tonemap_ACES(color);
    color = pow(color, 1.0f / 2.2f);

    return float4(color, 1.0f);
}