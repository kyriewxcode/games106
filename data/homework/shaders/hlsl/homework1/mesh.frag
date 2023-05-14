Texture2D textureColorMap : register(t0, space1);
Texture2D textureMetallicRoughness : register(t1, space1);
Texture2D textureNormalMap : register(t2, space1);
Texture2D textureEmissive : register(t3, space1);
Texture2D textureOcclusion : register(t4, space1);

SamplerState samplerColorMap : register(s0, space1);
SamplerState samplerMetallicRoughness : register(s1, space1);
SamplerState samplerNormalMap : register(s2, space1);
SamplerState samplerEmissive : register(s3, space1);
SamplerState samplerOcclusion : register(s4, space1);

struct UBO
{
    float4x4 projection; // 投影矩阵
    float4x4 view;       // 视图矩阵
    float3 lightColor;
    float lightIntensity;
    float4 lightPos;     // 灯光位置
    float3 cameraPos;    // 相机位置
};
cbuffer ubo : register(b0) { UBO ubo; }

struct PushConsts
{
    float4x4 model;  // 模型矩阵
    float3 albedo;   // 基本颜色
    float roughness; // 粗糙度
    float metallic;  // 金属度
};
[[vk::push_constant]] PushConsts material;

static const float PI = 3.14159265359;

struct VSOutput
{
    [[vk::location(0)]] float3 WorldPos : POSITION;
    [[vk::location(1)]] float3 Normal : NORMAL0;
    [[vk::location(2)]] float2 UV : TEXCOORD0;
};

float2 LightingFuncGGX_FV(float dotLH, float roughness)
{
    float alpha = roughness * roughness;

    // F
    float F_a, F_b;
    float dotLH5 = pow(1.0f - dotLH, 5);
    F_a = 1.0f;
    F_b = dotLH5;

    // V
    float vis;
    float k = alpha / 2.0f;
    float k2 = k * k;
    float invK2 = 1.0f - k2;
    vis = rcp(dotLH * dotLH * invK2 + k2);

    return float2(F_a * vis, F_b * vis);
}

float LightingFuncGGX_D(float dotNH, float roughness)
{
    float alpha = roughness * roughness;
    float alphaSqr = alpha * alpha;
    float pi = 3.14159f;
    float denom = dotNH * dotNH * (alphaSqr - 1.0) + 1.0f;

    float D = alphaSqr / (pi * denom * denom);
    return D;
}

float LightingFuncGGX(float3 N, float3 V, float3 L, float roughness, float F0)
{
    float3 H = normalize(V + L);

    float dotNL = saturate(dot(N, L));
    float dotLH = saturate(dot(L, H));
    float dotNH = saturate(dot(N, H));

    float D = LightingFuncGGX_D(dotNH, roughness);
    float2 FV_helper = LightingFuncGGX_FV(dotLH, roughness);
    float FV = F0 * FV_helper.x + (1.0f - F0) * FV_helper.y;
    float specular = dotNL * D * FV;

    return specular;
}

float4 main(VSOutput input) : SV_TARGET
{
    float3 textureColor = textureOcclusion.Sample(samplerOcclusion, input.UV).rgb;

    float3 N = normalize(input.Normal);
    float3 V = normalize(ubo.cameraPos - input.WorldPos);
    float3 L = normalize(ubo.lightPos.xyz - input.WorldPos);
    float3 specular = LightingFuncGGX(N, V, L, material.roughness, lerp(0.04, 1, material.metallic));

    float3 ambient = textureColor * float3(0.1, 0.1, 0.1) * material.albedo;
    float3 color = ambient + specular;
    color = pow(color, float3(0.4545, 0.4545, 0.4545));

    return float4(textureColor, 1.0f);
}