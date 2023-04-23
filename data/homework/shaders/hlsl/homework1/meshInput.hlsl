struct UBO
{
    float4x4 projection;
    float4x4 view;
    float4 lightPos;
    float4 viewPos;
};
cbuffer ubo : register(b0) { UBO ubo; }

struct PushConsts
{
    float4x4 modelMatrix;
    float3 baseColorFactor;
    float roughnessFactor;
    float3 emissiveFactor;
    float metallicFactor;
    float3 lightColor;
    float lightIntensity;
};
[[vk::push_constant]] PushConsts material;

#define PI 3.14159265359

struct VSInput
{
    [[vk::location(0)]] float3 Pos : POSITION0;
    [[vk::location(1)]] float3 Normal : NORMAL0;
    [[vk::location(2)]] float2 UV : TEXCOORD0;
    [[vk::location(3)]] float4 Tangent : TEXCOORD1;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    [[vk::location(1)]] float3 Normal : NORMAL0;
    [[vk::location(0)]] float3 WorldPos : TEXCOORD0;
    [[vk::location(2)]] float2 UV : TEXCOORD1;
    [[vk::location(3)]] float3 Tangent : TEXCOORD2;
};

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