# 作业1记录
1. 读取动画数据，实时更新modelMatrix
2. 添加了材质统一的descriptor，为没有Texture的Node添加dummyTexture
3. PBR PixelShader（D_GGX, G_Smith_GGX, F_Schlick）

由于工作繁忙，Tonemap并没有单独添加Pass，直接在PS里加的

结果如下:  

![image](Result.png)