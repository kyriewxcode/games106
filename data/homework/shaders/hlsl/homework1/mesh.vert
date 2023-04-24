// Copyright 2020 Google LLC

struct VSInput
{
	[[vk::location(0)]] float3 Pos : POSITION0;
	[[vk::location(1)]] float3 Normal : NORMAL0;
	[[vk::location(2)]] float2 UV : TEXCOORD0;
};

struct UBO
{
    float4x4 projection; // 投影矩阵
    float4x4 view;       // 视图矩阵
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

struct VSOutput
{
	float4 Pos : SV_POSITION;
	[[vk::location(0)]] float3 WorldPos : POSITION;
	[[vk::location(1)]] float3 Normal : NORMAL0;
	[[vk::location(2)]] float2 UV : TEXCOORD0;
};

VSOutput main(VSInput input)
{
	VSOutput output = (VSOutput)0;
	output.Normal = input.Normal;
	output.UV = input.UV;
	float4 WorldPos = mul(material.model, float4(input.Pos.xyz, 1.0));
	output.Pos = mul(ubo.projection, mul(ubo.view, WorldPos));
	output.WorldPos = WorldPos.rgb;
	output.Normal = mul((float3x3)ubo.view, input.Normal);
	return output;
}