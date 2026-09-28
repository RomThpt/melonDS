/*
    Copyright 2016-2026 melonDS team

    This file is part of melonDS.

    melonDS is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    melonDS is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with melonDS. If not, see http://www.gnu.org/licenses/.
*/

#include "GPU3D_Vulkan.h"

#include <cstring>
#include <string>

#include <shaderc/shaderc.h>

#include "GPU.h"
#include "GPU3D_Compute_shaders.h"
#include "Platform.h"

namespace melonDS
{

using Platform::Log;
using Platform::LogLevel;

namespace
{

struct MetaUniform
{
    u32 NumPolygons;
    u32 NumVariants;
    s32 AlphaRef;
    u32 DispCnt;
    u32 ToonTable[34][4];
    u32 ClearColor;
    u32 ClearDepth;
    u32 ClearAttr;
    u32 FogOffset;
    u32 FogShift;
    u32 FogColor;
    float ClearBitmapOffset[2];
};

static_assert(sizeof(MetaUniform) == 592);

void ReplaceAll(std::string& value, const std::string& from, const std::string& to)
{
    size_t position = 0;
    while ((position = value.find(from, position)) != std::string::npos)
    {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

std::string MakeVulkanShaderSource(const std::string& body,
                                   const std::vector<const char*>& defines)
{
    std::string source = "#version 450\n";
    for (const char* define : defines)
    {
        source += "#define ";
        source += define;
        source += '\n';
    }

    source += "#define ScreenWidth 256\n";
    source += "#define ScreenHeight 192\n";
    source += "#define MaxWorkTiles 12288\n";
    source += "#define TileSize 8\n";
    source += "const int CoarseTileCountY = 4;\n";
    source += "#define CoarseTileArea 32\n";
    source += "#define ClearCoarseBinMaskLocalSize 64\n";
    source += ComputeRendererShaders::Common;
    source += body;

    ReplaceAll(source, "layout (std430, binding =", "layout (std430, set = 0, binding =");
    ReplaceAll(source, "layout (std140, binding =", "layout (std140, set = 1, binding =");
    ReplaceAll(source, "layout (binding =", "layout (set = 2, binding =");

    const std::string looseUniforms =
        "layout (location = 0) uniform uint CurVariant;\n"
        "layout (location = 1) uniform vec2 InvTextureSize;\n"
        "layout (location = 2) uniform int TexIsCapture;\n"
        "layout (location = 3) uniform float CaptureYOffset;";
    const std::string pushConstants =
        "layout (push_constant) uniform RasterPushConstants\n"
        "{\n"
        "    uint CurVariant;\n"
        "    vec2 InvTextureSize;\n"
        "    int TexIsCapture;\n"
        "    float CaptureYOffset;\n"
        "};";
    ReplaceAll(source, looseUniforms, pushConstants);
    return source;
}

}

VulkanRenderer3D::VulkanRenderer3D(melonDS::GPU3D& gpu3D, Vulkan::Context& context) noexcept
    : Renderer3D(gpu3D), Context(context)
{
}

VulkanRenderer3D::~VulkanRenderer3D()
{
    DestroyResources();
}

bool VulkanRenderer3D::Init()
{
    VkDevice device = Context.GetDevice();

    VkCommandBufferAllocateInfo commandBufferInfo = {};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferInfo.commandPool = Context.GetCommandPool();
    commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferInfo.commandBufferCount = 1;
    VkResult result = vkAllocateCommandBuffers(device, &commandBufferInfo, &CommandBuffer);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to allocate command buffer: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = vkCreateFence(device, &fenceInfo, nullptr, &Fence);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to create fence: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    if (!CreateColorImage() || !CreateReadbackBuffer())
        return false;
    Log(LogLevel::Info, "Vulkan: compiling compute shaders\n");
    if (!CompileShaders())
        return false;
    Log(LogLevel::Info, "Vulkan: creating compute pipelines\n");
    if (!CreateComputePipelines())
        return false;
    Log(LogLevel::Info, "Vulkan: creating compute resources\n");
    return CreateComputeResources();
}

bool VulkanRenderer3D::CompileShader(const std::string& source,
                                     const std::vector<const char*>& defines,
                                     const char* name)
{
    const std::string shaderSource = MakeVulkanShaderSource(source, defines);
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    shaderc_compile_options_t options = shaderc_compile_options_initialize();
    if (!compiler || !options)
    {
        if (options) shaderc_compile_options_release(options);
        if (compiler) shaderc_compiler_release(compiler);
        return false;
    }

    shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan,
                                            shaderc_env_version_vulkan_1_1);
    shaderc_compile_options_set_target_spirv(options, shaderc_spirv_version_1_3);
    shaderc_compile_options_set_optimization_level(options,
                                                    shaderc_optimization_level_performance);
    shaderc_compilation_result_t result = shaderc_compile_into_spv(
        compiler, shaderSource.c_str(), shaderSource.size(), shaderc_compute_shader,
        name, "main", options);

    const shaderc_compilation_status status = shaderc_result_get_compilation_status(result);
    if (status != shaderc_compilation_status_success)
    {
        Log(LogLevel::Error, "Vulkan: failed to compile %s:\n%s\n", name,
            shaderc_result_get_error_message(result));
        shaderc_result_release(result);
        shaderc_compile_options_release(options);
        shaderc_compiler_release(compiler);
        return false;
    }

    VkShaderModuleCreateInfo moduleInfo = {};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = shaderc_result_get_length(result);
    moduleInfo.pCode = reinterpret_cast<const u32*>(shaderc_result_get_bytes(result));
    VkShaderModule module = VK_NULL_HANDLE;
    const VkResult moduleResult = vkCreateShaderModule(Context.GetDevice(), &moduleInfo,
                                                        nullptr, &module);
    shaderc_result_release(result);
    shaderc_compile_options_release(options);
    shaderc_compiler_release(compiler);
    if (moduleResult != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to create %s module: %s (%d)\n", name,
            Vulkan::ResultName(moduleResult), moduleResult);
        return false;
    }
    ShaderModules.push_back(module);
    return true;
}

bool VulkanRenderer3D::CompileShaders()
{
    using namespace ComputeRendererShaders;

    return
        CompileShader(InterpSpans, {"InterpSpans", "ZBuffer"}, "InterpSpansZ") &&
        CompileShader(InterpSpans, {"InterpSpans", "WBuffer"}, "InterpSpansW") &&
        CompileShader(BinCombined, {"BinCombined"}, "BinCombined") &&
        CompileShader(DepthBlend, {"DepthBlend", "ZBuffer"}, "DepthBlendZ") &&
        CompileShader(DepthBlend, {"DepthBlend", "WBuffer"}, "DepthBlendW") &&
        CompileShader(Rasterise, {"Rasterise", "ZBuffer", "NoTexture"}, "RasterNoTextureZ") &&
        CompileShader(Rasterise, {"Rasterise", "WBuffer", "NoTexture"}, "RasterNoTextureW") &&
        CompileShader(Rasterise, {"Rasterise", "ZBuffer", "NoTexture", "Toon"}, "RasterNoTextureToonZ") &&
        CompileShader(Rasterise, {"Rasterise", "WBuffer", "NoTexture", "Toon"}, "RasterNoTextureToonW") &&
        CompileShader(Rasterise, {"Rasterise", "ZBuffer", "NoTexture", "Highlight"}, "RasterNoTextureHighlightZ") &&
        CompileShader(Rasterise, {"Rasterise", "WBuffer", "NoTexture", "Highlight"}, "RasterNoTextureHighlightW") &&
        CompileShader(Rasterise, {"Rasterise", "ZBuffer", "UseTexture", "Decal"}, "RasterTextureDecalZ") &&
        CompileShader(Rasterise, {"Rasterise", "WBuffer", "UseTexture", "Decal"}, "RasterTextureDecalW") &&
        CompileShader(Rasterise, {"Rasterise", "ZBuffer", "UseTexture", "Modulate"}, "RasterTextureModulateZ") &&
        CompileShader(Rasterise, {"Rasterise", "WBuffer", "UseTexture", "Modulate"}, "RasterTextureModulateW") &&
        CompileShader(Rasterise, {"Rasterise", "ZBuffer", "UseTexture", "Toon"}, "RasterTextureToonZ") &&
        CompileShader(Rasterise, {"Rasterise", "WBuffer", "UseTexture", "Toon"}, "RasterTextureToonW") &&
        CompileShader(Rasterise, {"Rasterise", "ZBuffer", "UseTexture", "Highlight"}, "RasterTextureHighlightZ") &&
        CompileShader(Rasterise, {"Rasterise", "WBuffer", "UseTexture", "Highlight"}, "RasterTextureHighlightW") &&
        CompileShader(Rasterise, {"Rasterise", "ZBuffer", "ShadowMask"}, "RasterShadowZ") &&
        CompileShader(Rasterise, {"Rasterise", "WBuffer", "ShadowMask"}, "RasterShadowW") &&
        CompileShader(ClearCoarseBinMask, {"ClearCoarseBinMask"}, "ClearCoarseBinMask") &&
        CompileShader(ClearIndirectWorkCount, {"ClearIndirectWorkCount"}, "ClearIndirectWorkCount") &&
        CompileShader(CalcOffsets, {"CalculateWorkOffsets"}, "CalculateWorkOffsets") &&
        CompileShader(SortWork, {"SortWork"}, "SortWork") &&
        CompileShader(FinalPass, {"FinalPass"}, "FinalPass") &&
        CompileShader(FinalPass, {"FinalPass", "EdgeMarking"}, "FinalPassEdge") &&
        CompileShader(FinalPass, {"FinalPass", "Fog"}, "FinalPassFog") &&
        CompileShader(FinalPass, {"FinalPass", "EdgeMarking", "Fog"}, "FinalPassEdgeFog") &&
        CompileShader(FinalPass, {"FinalPass", "AntiAliasing"}, "FinalPassAA") &&
        CompileShader(FinalPass, {"FinalPass", "AntiAliasing", "EdgeMarking"}, "FinalPassAAEdge") &&
        CompileShader(FinalPass, {"FinalPass", "AntiAliasing", "Fog"}, "FinalPassAAFog") &&
        CompileShader(FinalPass, {"FinalPass", "AntiAliasing", "EdgeMarking", "Fog"}, "FinalPassAAEdgeFog");
}

bool VulkanRenderer3D::CreateDescriptorSetLayout(
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    VkDescriptorSetLayout& layout)
{
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = static_cast<u32>(bindings.size());
    info.pBindings = bindings.data();
    const VkResult result = vkCreateDescriptorSetLayout(Context.GetDevice(), &info, nullptr,
                                                         &layout);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to create descriptor layout: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }
    return true;
}

bool VulkanRenderer3D::CreatePipelineLayout(
    const std::vector<VkDescriptorSetLayout>& setLayouts, bool pushConstants,
    VkPipelineLayout& layout)
{
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.size = 24;

    VkPipelineLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = static_cast<u32>(setLayouts.size());
    info.pSetLayouts = setLayouts.data();
    info.pushConstantRangeCount = pushConstants ? 1 : 0;
    info.pPushConstantRanges = pushConstants ? &pushConstantRange : nullptr;
    const VkResult result = vkCreatePipelineLayout(Context.GetDevice(), &info, nullptr, &layout);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to create pipeline layout: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }
    return true;
}

bool VulkanRenderer3D::CreateComputePipelines()
{
    std::vector<VkDescriptorSetLayoutBinding> bufferBindings(8);
    for (u32 i = 0; i < bufferBindings.size(); i++)
    {
        bufferBindings[i].binding = i;
        bufferBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bufferBindings[i].descriptorCount = 1;
        bufferBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    if (!CreateDescriptorSetLayout(bufferBindings, BufferSetLayout)) return false;

    VkDescriptorSetLayoutBinding uniformBinding = {};
    uniformBinding.binding = 0;
    uniformBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBinding.descriptorCount = 1;
    uniformBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    if (!CreateDescriptorSetLayout({uniformBinding}, UniformSetLayout)) return false;

    VkDescriptorSetLayoutBinding spanBinding = {};
    spanBinding.binding = 0;
    spanBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    spanBinding.descriptorCount = 1;
    spanBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    if (!CreateDescriptorSetLayout({spanBinding}, SpanImageSetLayout)) return false;

    std::vector<VkDescriptorSetLayoutBinding> textureBindings(3);
    for (u32 i = 0; i < textureBindings.size(); i++)
    {
        textureBindings[i].binding = i;
        textureBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textureBindings[i].descriptorCount = 1;
        textureBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    if (!CreateDescriptorSetLayout(textureBindings, TextureSetLayout)) return false;

    VkDescriptorSetLayoutBinding outputBinding = {};
    outputBinding.binding = 0;
    outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outputBinding.descriptorCount = 1;
    outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    if (!CreateDescriptorSetLayout({outputBinding}, OutputImageSetLayout)) return false;

    if (!CreatePipelineLayout({BufferSetLayout, UniformSetLayout}, false,
                              BasePipelineLayout)) return false;
    if (!CreatePipelineLayout({BufferSetLayout, UniformSetLayout, SpanImageSetLayout}, false,
                              SpanPipelineLayout)) return false;
    if (!CreatePipelineLayout({BufferSetLayout, UniformSetLayout, TextureSetLayout}, true,
                              RasterPipelineLayout)) return false;
    if (!CreatePipelineLayout({BufferSetLayout, UniformSetLayout, TextureSetLayout}, false,
                              TexturePipelineLayout)) return false;
    if (!CreatePipelineLayout({BufferSetLayout, UniformSetLayout, OutputImageSetLayout}, false,
                              OutputPipelineLayout)) return false;

    ComputePipelines.reserve(ShaderModules.size());
    for (u32 i = 0; i < ShaderModules.size(); i++)
    {
        VkPipelineLayout layout = BasePipelineLayout;
        if (i <= 1)
            layout = SpanPipelineLayout;
        else if (i >= 5 && i <= 20)
            layout = RasterPipelineLayout;
        else if (i == 3 || i == 4)
            layout = TexturePipelineLayout;
        else if (i >= 25)
            layout = OutputPipelineLayout;

        VkPipelineShaderStageCreateInfo stage = {};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = ShaderModules[i];
        stage.pName = "main";

        VkComputePipelineCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.stage = stage;
        info.layout = layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult result = vkCreateComputePipelines(Context.GetDevice(), VK_NULL_HANDLE,
                                                          1, &info, nullptr, &pipeline);
        if (result != VK_SUCCESS)
        {
            Log(LogLevel::Error, "Vulkan: failed to create compute pipeline %u: %s (%d)\n",
                i, Vulkan::ResultName(result), result);
            return false;
        }
        ComputePipelines.push_back(pipeline);
    }
    return true;
}

bool VulkanRenderer3D::CreateBuffer(Buffer& buffer, VkDeviceSize size,
                                    VkBufferUsageFlags usage)
{
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(Context.GetDevice(), &info, nullptr, &buffer.Handle);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to create buffer: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    VkMemoryRequirements requirements = {};
    vkGetBufferMemoryRequirements(Context.GetDevice(), buffer.Handle, &requirements);
    u32 memoryType = 0;
    constexpr VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!Context.FindMemoryType(requirements.memoryTypeBits, properties, memoryType))
    {
        Log(LogLevel::Error, "Vulkan: no host-visible coherent memory for buffer\n");
        return false;
    }

    VkMemoryAllocateInfo allocation = {};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType;
    result = vkAllocateMemory(Context.GetDevice(), &allocation, nullptr, &buffer.Memory);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to allocate buffer memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }
    result = vkBindBufferMemory(Context.GetDevice(), buffer.Handle, buffer.Memory, 0);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to bind buffer memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }
    result = vkMapMemory(Context.GetDevice(), buffer.Memory, 0, size, 0, &buffer.Mapped);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to map buffer memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }
    buffer.Size = size;
    std::memset(buffer.Mapped, 0, size);
    return true;
}

bool VulkanRenderer3D::CreateImage(u32 width, u32 height, VkFormat format,
                                   VkImageUsageFlags usage, VkImage& image,
                                   VkDeviceMemory& memory, VkImageView& view,
                                   VkImageViewType viewType, u32 layers)
{
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = layers;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkResult result = vkCreateImage(Context.GetDevice(), &imageInfo, nullptr, &image);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to create image: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    VkMemoryRequirements requirements = {};
    vkGetImageMemoryRequirements(Context.GetDevice(), image, &requirements);
    u32 memoryType = 0;
    if (!Context.FindMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                memoryType))
    {
        Log(LogLevel::Error, "Vulkan: no device-local memory for image\n");
        return false;
    }
    VkMemoryAllocateInfo allocation = {};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType;
    result = vkAllocateMemory(Context.GetDevice(), &allocation, nullptr, &memory);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to allocate image memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }
    result = vkBindImageMemory(Context.GetDevice(), image, memory, 0);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to bind image memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = viewType;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = layers;
    result = vkCreateImageView(Context.GetDevice(), &viewInfo, nullptr, &view);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to create image view: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }
    return true;
}

bool VulkanRenderer3D::CreateComputeResources()
{
    constexpr VkBufferUsageFlags storageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    constexpr VkDeviceSize polygonSize = 2048 * 40;
    constexpr VkDeviceSize xSpanSize = MaxYSpanIndices * 96;
    constexpr VkDeviceSize ySpanSize = MaxYSpanSetups * 124;
    constexpr VkDeviceSize setupIndicesSize = MaxYSpanIndices * 8;
    constexpr VkDeviceSize tileSize = 4 * TileSize * TileSize * MaxWorkTiles;
    constexpr VkDeviceSize resultSize = 4 * 3 * 2 * ScreenWidth * ScreenHeight;
    constexpr VkDeviceSize binSize = MaxVariants * 16 + MaxVariants * 4 + 16 +
        TilesPerLine * TileLines * (CoarseBinStride + BinStride * 2) * 4;
    constexpr VkDeviceSize workSize = MaxWorkTiles * 2 * 4 * 2;
    constexpr VkDeviceSize metaSize = 592;
    constexpr VkDeviceSize clearUploadSize = 2 * 256 * 256 * sizeof(u32);

    if (!CreateBuffer(PolygonBuffer, polygonSize, storageUsage) ||
        !CreateBuffer(XSpanBuffer, xSpanSize, storageUsage) ||
        !CreateBuffer(YSpanBuffer, ySpanSize, storageUsage) ||
        !CreateBuffer(SetupIndicesBuffer, setupIndicesSize,
                      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) ||
        !CreateBuffer(ColorTileBuffer, tileSize, storageUsage) ||
        !CreateBuffer(DepthTileBuffer, tileSize, storageUsage) ||
        !CreateBuffer(AttrTileBuffer, tileSize, storageUsage) ||
        !CreateBuffer(ResultBuffer, resultSize, storageUsage) ||
        !CreateBuffer(BinResultBuffer, binSize,
                      storageUsage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) ||
        !CreateBuffer(WorkBuffer, workSize, storageUsage) ||
        !CreateBuffer(MetaBuffer, metaSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
        !CreateBuffer(ClearUploadBuffer, clearUploadSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        return false;

    VkBufferViewCreateInfo bufferViewInfo = {};
    bufferViewInfo.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    bufferViewInfo.buffer = SetupIndicesBuffer.Handle;
    bufferViewInfo.format = VK_FORMAT_R16G16B16A16_UINT;
    bufferViewInfo.range = SetupIndicesBuffer.Size;
    VkResult result = vkCreateBufferView(Context.GetDevice(), &bufferViewInfo, nullptr,
                                         &SetupIndicesView);
    if (result != VK_SUCCESS)
        return false;

    for (u32 i = 0; i < 2; i++)
    {
        if (!CreateImage(256, 256, VK_FORMAT_R32_UINT,
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         ClearImages[i], ClearImageMemory[i], ClearImageViews[i]))
            return false;
    }

    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.maxLod = 0.0f;
    result = vkCreateSampler(Context.GetDevice(), &samplerInfo, nullptr, &ClearSampler);
    if (result != VK_SUCCESS)
        return false;

    std::array<VkDescriptorPoolSize, 5> poolSizes = {{
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    }};
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 6;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    result = vkCreateDescriptorPool(Context.GetDevice(), &poolInfo, nullptr, &DescriptorPool);
    if (result != VK_SUCCESS)
        return false;

    std::array<VkDescriptorSetLayout, 6> layouts = {
        BufferSetLayout, BufferSetLayout, UniformSetLayout, SpanImageSetLayout,
        TextureSetLayout, OutputImageSetLayout};
    std::array<VkDescriptorSet, 6> sets = {};
    VkDescriptorSetAllocateInfo setInfo = {};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = DescriptorPool;
    setInfo.descriptorSetCount = layouts.size();
    setInfo.pSetLayouts = layouts.data();
    result = vkAllocateDescriptorSets(Context.GetDevice(), &setInfo, sets.data());
    if (result != VK_SUCCESS)
        return false;
    SpanBufferSet = sets[0];
    RasterBufferSet = sets[1];
    UniformSet = sets[2];
    SpanImageSet = sets[3];
    ClearTextureSet = sets[4];
    OutputImageSet = sets[5];

    const std::array<Buffer*, 8> spanBuffers = {&PolygonBuffer, &XSpanBuffer, &YSpanBuffer,
        &DepthTileBuffer, &AttrTileBuffer, &ResultBuffer, &BinResultBuffer, &WorkBuffer};
    const std::array<Buffer*, 8> rasterBuffers = {&PolygonBuffer, &XSpanBuffer,
        &ColorTileBuffer, &DepthTileBuffer, &AttrTileBuffer, &ResultBuffer,
        &BinResultBuffer, &WorkBuffer};
    std::array<VkDescriptorBufferInfo, 16> bufferInfos = {};
    std::array<VkWriteDescriptorSet, 22> writes = {};
    u32 writeCount = 0;
    for (u32 set = 0; set < 2; set++)
    {
        const auto& buffers = set == 0 ? spanBuffers : rasterBuffers;
        for (u32 binding = 0; binding < buffers.size(); binding++)
        {
            const u32 index = set * 8 + binding;
            bufferInfos[index] = {buffers[binding]->Handle, 0, buffers[binding]->Size};
            writes[writeCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[writeCount].dstSet = set == 0 ? SpanBufferSet : RasterBufferSet;
            writes[writeCount].dstBinding = binding;
            writes[writeCount].descriptorCount = 1;
            writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[writeCount].pBufferInfo = &bufferInfos[index];
            writeCount++;
        }
    }
    VkDescriptorBufferInfo uniformInfo = {MetaBuffer.Handle, 0, MetaBuffer.Size};
    writes[writeCount] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[writeCount].dstSet = UniformSet;
    writes[writeCount].descriptorCount = 1;
    writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[writeCount].pBufferInfo = &uniformInfo;
    writeCount++;
    writes[writeCount] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[writeCount].dstSet = SpanImageSet;
    writes[writeCount].descriptorCount = 1;
    writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    writes[writeCount].pTexelBufferView = &SetupIndicesView;
    writeCount++;
    std::array<VkDescriptorImageInfo, 3> clearInfos = {};
    for (u32 i = 0; i < clearInfos.size(); i++)
    {
        clearInfos[i].sampler = ClearSampler;
        clearInfos[i].imageView = ClearImageViews[std::min(i, 1U)];
        clearInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[writeCount] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[writeCount].dstSet = ClearTextureSet;
        writes[writeCount].dstBinding = i;
        writes[writeCount].descriptorCount = 1;
        writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeCount].pImageInfo = &clearInfos[i];
        writeCount++;
    }
    VkDescriptorImageInfo outputInfo = {};
    outputInfo.imageView = ColorView;
    outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    writes[writeCount] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[writeCount].dstSet = OutputImageSet;
    writes[writeCount].descriptorCount = 1;
    writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[writeCount].pImageInfo = &outputInfo;
    writeCount++;
    vkUpdateDescriptorSets(Context.GetDevice(), writeCount, writes.data(), 0, nullptr);
    return true;
}

bool VulkanRenderer3D::CreateColorImage()
{
    return CreateImage(ScreenWidth, ScreenHeight, VK_FORMAT_R8G8B8A8_UNORM,
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                           VK_IMAGE_USAGE_STORAGE_BIT,
                       ColorImage, ColorMemory, ColorView);
}

bool VulkanRenderer3D::CreateReadbackBuffer()
{
    VkDevice device = Context.GetDevice();

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = FramebufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &ReadbackBuffer);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to create readback buffer: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    VkMemoryRequirements requirements = {};
    vkGetBufferMemoryRequirements(device, ReadbackBuffer, &requirements);
    u32 memoryType = 0;
    const VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!Context.FindMemoryType(requirements.memoryTypeBits, properties, memoryType))
    {
        Log(LogLevel::Error, "Vulkan: no host-visible memory for readback buffer\n");
        return false;
    }

    VkMemoryAllocateInfo allocationInfo = {};
    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.allocationSize = requirements.size;
    allocationInfo.memoryTypeIndex = memoryType;
    result = vkAllocateMemory(device, &allocationInfo, nullptr, &ReadbackMemory);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to allocate readback memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    result = vkBindBufferMemory(device, ReadbackBuffer, ReadbackMemory, 0);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to bind readback memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    result = vkMapMemory(device, ReadbackMemory, 0, FramebufferSize, 0,
                         reinterpret_cast<void**>(&Framebuffer));
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to map readback memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        Framebuffer = nullptr;
        return false;
    }
    std::memset(Framebuffer, 0, FramebufferSize);
    return true;
}

void VulkanRenderer3D::Reset()
{
    if (Framebuffer)
        std::memset(Framebuffer, 0, FramebufferSize);
}

void VulkanRenderer3D::SetRenderSettings(int scale) noexcept
{
    (void)scale;
}

void VulkanRenderer3D::PrepareBitmapClear()
{
    auto* output = static_cast<u32*>(ClearUploadBuffer.Mapped);
    u32* colorOutput = output;
    u32* depthOutput = output + 256 * 256;
    for (u32 i = 0; i < 256 * 256; i++)
    {
        const u16 colorValue = GPU.ReadVRAMFlat_Texture<u16>(0x40000 + i * 2);
        const u16 depthValue = GPU.ReadVRAMFlat_Texture<u16>(0x60000 + i * 2);
        u32 red = (colorValue << 1) & 0x3E;
        u32 green = (colorValue >> 4) & 0x3E;
        u32 blue = (colorValue >> 9) & 0x3E;
        if (red) red++;
        if (green) green++;
        if (blue) blue++;
        const u32 alpha = (colorValue & 0x8000) ? 0x1F : 0;
        colorOutput[i] = red | (green << 8) | (blue << 16) | (alpha << 24);
        depthOutput[i] = ((depthValue & 0x7FFF) * 0x200) + 0x1FF |
                         ((depthValue & 0x8000) << 9);
    }
}

bool VulkanRenderer3D::SubmitFrame(bool bitmapClear)
{
    VkDevice device = Context.GetDevice();

    MetaUniform meta = {};
    meta.AlphaRef = GPU3D.RenderAlphaRef;
    meta.DispCnt = GPU3D.RenderDispCnt;
    {
        u32 red = (GPU3D.RenderClearAttr1 << 1) & 0x3E;
        u32 green = (GPU3D.RenderClearAttr1 >> 4) & 0x3E;
        u32 blue = (GPU3D.RenderClearAttr1 >> 9) & 0x3E;
        if (red) red++;
        if (green) green++;
        if (blue) blue++;
        meta.ClearColor = red | (green << 8) | (blue << 16) |
                          (((GPU3D.RenderClearAttr1 >> 16) & 0x1F) << 24);
        meta.ClearDepth = ((GPU3D.RenderClearAttr2 & 0x7FFF) * 0x200) + 0x1FF;
        meta.ClearAttr = GPU3D.RenderClearAttr1 & 0x3F008000;
        meta.ClearBitmapOffset[0] = float((GPU3D.RenderClearAttr2 >> 16) & 0xFF) / 256.0f;
        meta.ClearBitmapOffset[1] = float((GPU3D.RenderClearAttr2 >> 24) & 0xFF) / 256.0f;
    }
    for (u32 i = 0; i < 32; i++)
    {
        const u32 color = GPU3D.RenderToonTable[i];
        u32 red = (color << 1) & 0x3E;
        u32 green = (color >> 4) & 0x3E;
        u32 blue = (color >> 9) & 0x3E;
        if (red) red++;
        if (green) green++;
        if (blue) blue++;
        meta.ToonTable[i][0] = red | (green << 8) | (blue << 16);
    }
    for (u32 i = 0; i < 34; i++)
        meta.ToonTable[i][1] = GPU3D.RenderFogDensityTable[i];
    for (u32 i = 0; i < 8; i++)
    {
        const u32 color = GPU3D.RenderEdgeTable[i];
        u32 red = (color << 1) & 0x3E;
        u32 green = (color >> 4) & 0x3E;
        u32 blue = (color >> 9) & 0x3E;
        if (red) red++;
        if (green) green++;
        if (blue) blue++;
        meta.ToonTable[i][2] = red | (green << 8) | (blue << 16);
    }
    meta.FogOffset = GPU3D.RenderFogOffset;
    meta.FogShift = GPU3D.RenderFogShift;
    {
        const u32 color = GPU3D.RenderFogColor;
        u32 red = (color << 1) & 0x3E;
        u32 green = (color >> 4) & 0x3E;
        u32 blue = (color >> 9) & 0x3E;
        if (red) red++;
        if (green) green++;
        if (blue) blue++;
        meta.FogColor = red | (green << 8) | (blue << 16) |
                        (((color >> 16) & 0x1F) << 24);
    }
    std::memcpy(MetaBuffer.Mapped, &meta, sizeof(meta));

    VkResult result = vkResetCommandBuffer(CommandBuffer, 0);
    if (result != VK_SUCCESS)
        return false;

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(CommandBuffer, &beginInfo);
    if (result != VK_SUCCESS)
        return false;

    VkMemoryBarrier hostWriteBarrier = {};
    hostWriteBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hostWriteBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    hostWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                     VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &hostWriteBarrier, 0, nullptr, 0, nullptr);

    if (bitmapClear || ClearImageLayouts[0] == VK_IMAGE_LAYOUT_UNDEFINED)
    {
        for (u32 i = 0; i < 2; i++)
        {
            VkImageMemoryBarrier toTransfer = {};
            toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toTransfer.srcAccessMask = ClearImageLayouts[i] == VK_IMAGE_LAYOUT_UNDEFINED ? 0 :
                                       VK_ACCESS_SHADER_READ_BIT;
            toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toTransfer.oldLayout = ClearImageLayouts[i];
            toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.image = ClearImages[i];
            toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toTransfer.subresourceRange.levelCount = 1;
            toTransfer.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(CommandBuffer,
                ClearImageLayouts[i] == VK_IMAGE_LAYOUT_UNDEFINED ?
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransfer);

            VkBufferImageCopy upload = {};
            upload.bufferOffset = i * 256 * 256 * sizeof(u32);
            upload.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            upload.imageSubresource.layerCount = 1;
            upload.imageExtent = {256, 256, 1};
            vkCmdCopyBufferToImage(CommandBuffer, ClearUploadBuffer.Handle, ClearImages[i],
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload);

            VkImageMemoryBarrier toSample = toTransfer;
            toSample.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toSample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            toSample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toSample.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &toSample);
            ClearImageLayouts[i] = VK_IMAGE_LAYOUT_GENERAL;
        }
    }

    VkImageMemoryBarrier toGeneral = {};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.srcAccessMask = ColorLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 :
                              VK_ACCESS_TRANSFER_READ_BIT;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toGeneral.oldLayout = ColorLayout;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = ColorImage;
    toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneral.subresourceRange.levelCount = 1;
    toGeneral.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(CommandBuffer,
        ColorLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT :
                                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);

    const VkDescriptorSet baseSets[] = {RasterBufferSet, UniformSet};
    vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelines[21]);
    vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            BasePipelineLayout, 0, 2, baseSets, 0, nullptr);
    vkCmdDispatch(CommandBuffer, TilesPerLine * TileLines / 64, 1, 1);

    VkMemoryBarrier computeBarrier = {};
    computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier,
                         0, nullptr, 0, nullptr);

    const VkDescriptorSet textureSets[] = {RasterBufferSet, UniformSet, ClearTextureSet};
    vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipelines[3]);
    vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            TexturePipelineLayout, 0, 3, textureSets, 0, nullptr);
    vkCmdDispatch(CommandBuffer, TilesPerLine, TileLines, 1);
    vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier,
                         0, nullptr, 0, nullptr);

    u32 finalPass = 0;
    if (GPU3D.RenderDispCnt & (1 << 4)) finalPass |= 4;
    if (GPU3D.RenderDispCnt & (1 << 7)) finalPass |= 2;
    if (GPU3D.RenderDispCnt & (1 << 5)) finalPass |= 1;
    const VkDescriptorSet outputSets[] = {RasterBufferSet, UniformSet, OutputImageSet};
    vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      ComputePipelines[25 + finalPass]);
    vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            OutputPipelineLayout, 0, 3, outputSets, 0, nullptr);
    vkCmdDispatch(CommandBuffer, ScreenWidth / 32, ScreenHeight, 1);

    VkImageMemoryBarrier toTransferSource = toGeneral;
    toTransferSource.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toTransferSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toTransferSource.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toTransferSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &toTransferSource);

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = {ScreenWidth, ScreenHeight, 1};
    vkCmdCopyImageToBuffer(CommandBuffer, ColorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           ReadbackBuffer, 1, &copyRegion);

    VkMemoryBarrier hostReadBarrier = {};
    hostReadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hostReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    hostReadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hostReadBarrier,
                         0, nullptr, 0, nullptr);

    result = vkEndCommandBuffer(CommandBuffer);
    if (result != VK_SUCCESS)
        return false;

    result = vkResetFences(device, 1, &Fence);
    if (result != VK_SUCCESS)
        return false;

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &CommandBuffer;
    result = vkQueueSubmit(Context.GetQueue(), 1, &submitInfo, Fence);
    if (result != VK_SUCCESS)
        return false;

    result = vkWaitForFences(device, 1, &Fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS)
        return false;

    ColorLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    return true;
}

void VulkanRenderer3D::RenderFrame()
{
    auto textureDirty = GPU.VRAMDirty_Texture.DeriveState(GPU.VRAMMap_Texture, GPU);
    GPU.MakeVRAMFlat_TextureCoherent(textureDirty);

    const bool bitmapClear = (GPU3D.RenderDispCnt & (1 << 14)) != 0;
    if (bitmapClear)
        PrepareBitmapClear();

    if (!SubmitFrame(bitmapClear))
        Log(LogLevel::Error, "Vulkan: failed to submit 3D frame\n");
}

u32* VulkanRenderer3D::GetLine(int line)
{
    if (GPU3D.AbortFrame || !Framebuffer)
    {
        ScrolledLine.fill(0);
        return ScrolledLine.data();
    }

    u32* rawLine = &Framebuffer[line * ScreenWidth];
    const u16 xPosition = GPU3D.RenderXPos;
    if (xPosition == 0)
        return rawLine;

    if (xPosition & 0x100)
    {
        int destination = 0;
        int source = xPosition;
        for (; source < 512; destination++, source++)
            ScrolledLine[destination] = 0;
        for (source = 0; destination < static_cast<int>(ScreenWidth); destination++, source++)
            ScrolledLine[destination] = rawLine[source];
    }
    else
    {
        int destination = 0;
        int source = xPosition;
        for (; source < static_cast<int>(ScreenWidth); destination++, source++)
            ScrolledLine[destination] = rawLine[source];
        for (; destination < static_cast<int>(ScreenWidth); destination++)
            ScrolledLine[destination] = 0;
    }
    return ScrolledLine.data();
}

void VulkanRenderer3D::DestroyBuffer(Buffer& buffer)
{
    if (buffer.Mapped)
        vkUnmapMemory(Context.GetDevice(), buffer.Memory);
    if (buffer.Handle != VK_NULL_HANDLE)
        vkDestroyBuffer(Context.GetDevice(), buffer.Handle, nullptr);
    if (buffer.Memory != VK_NULL_HANDLE)
        vkFreeMemory(Context.GetDevice(), buffer.Memory, nullptr);
    buffer = {};
}

void VulkanRenderer3D::DestroyResources()
{
    VkDevice device = Context.GetDevice();
    if (device == VK_NULL_HANDLE)
        return;

    vkDeviceWaitIdle(device);
    if (DescriptorPool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, DescriptorPool, nullptr);
    if (ClearSampler != VK_NULL_HANDLE)
        vkDestroySampler(device, ClearSampler, nullptr);
    if (SetupIndicesView != VK_NULL_HANDLE)
        vkDestroyBufferView(device, SetupIndicesView, nullptr);
    for (u32 i = 0; i < 2; i++)
    {
        if (ClearImageViews[i] != VK_NULL_HANDLE)
            vkDestroyImageView(device, ClearImageViews[i], nullptr);
        if (ClearImages[i] != VK_NULL_HANDLE)
            vkDestroyImage(device, ClearImages[i], nullptr);
        if (ClearImageMemory[i] != VK_NULL_HANDLE)
            vkFreeMemory(device, ClearImageMemory[i], nullptr);
    }
    DestroyBuffer(ClearUploadBuffer);
    DestroyBuffer(MetaBuffer);
    DestroyBuffer(WorkBuffer);
    DestroyBuffer(BinResultBuffer);
    DestroyBuffer(ResultBuffer);
    DestroyBuffer(AttrTileBuffer);
    DestroyBuffer(DepthTileBuffer);
    DestroyBuffer(ColorTileBuffer);
    DestroyBuffer(SetupIndicesBuffer);
    DestroyBuffer(YSpanBuffer);
    DestroyBuffer(XSpanBuffer);
    DestroyBuffer(PolygonBuffer);
    if (Framebuffer)
        vkUnmapMemory(device, ReadbackMemory);
    if (ReadbackBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, ReadbackBuffer, nullptr);
    if (ReadbackMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, ReadbackMemory, nullptr);
    if (ColorView != VK_NULL_HANDLE)
        vkDestroyImageView(device, ColorView, nullptr);
    if (ColorImage != VK_NULL_HANDLE)
        vkDestroyImage(device, ColorImage, nullptr);
    if (ColorMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, ColorMemory, nullptr);
    if (Fence != VK_NULL_HANDLE)
        vkDestroyFence(device, Fence, nullptr);
    if (CommandBuffer != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device, Context.GetCommandPool(), 1, &CommandBuffer);
    for (VkPipeline pipeline : ComputePipelines)
        vkDestroyPipeline(device, pipeline, nullptr);
    ComputePipelines.clear();
    if (OutputPipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, OutputPipelineLayout, nullptr);
    if (TexturePipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, TexturePipelineLayout, nullptr);
    if (RasterPipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, RasterPipelineLayout, nullptr);
    if (SpanPipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, SpanPipelineLayout, nullptr);
    if (BasePipelineLayout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, BasePipelineLayout, nullptr);
    if (OutputImageSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, OutputImageSetLayout, nullptr);
    if (TextureSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, TextureSetLayout, nullptr);
    if (SpanImageSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, SpanImageSetLayout, nullptr);
    if (UniformSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, UniformSetLayout, nullptr);
    if (BufferSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, BufferSetLayout, nullptr);
    for (VkShaderModule module : ShaderModules)
        vkDestroyShaderModule(device, module, nullptr);
    ShaderModules.clear();

    Framebuffer = nullptr;
    ReadbackBuffer = VK_NULL_HANDLE;
    ReadbackMemory = VK_NULL_HANDLE;
    ColorImage = VK_NULL_HANDLE;
    ColorMemory = VK_NULL_HANDLE;
    Fence = VK_NULL_HANDLE;
    CommandBuffer = VK_NULL_HANDLE;
}

}
