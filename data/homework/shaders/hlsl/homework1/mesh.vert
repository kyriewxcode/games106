#include "meshInput.hlsl"

VSOutput main(VSInput input)
{
    VSOutput output = (VSOutput)0;
    output.WorldPos = mul(material.modelMatrix, float4(input.Pos.xyz, 1.0)).xyz;
    output.Pos = mul(ubo.projection, mul(ubo.view, float4(output.WorldPos,1.0f)));
    output.Normal = mul((float3x3)material.modelMatrix, input.Normal);
    output.UV = input.UV;
    output.Tangent = mul((float3x3)material.modelMatrix, input.Tangent.xyz);
    return output;
}