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

    return CreateColorImage() && CreateReadbackBuffer() && CompileShaders();
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

bool VulkanRenderer3D::CreateColorImage()
{
    VkDevice device = Context.GetDevice();

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UINT;
    imageInfo.extent = {ScreenWidth, ScreenHeight, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkResult result = vkCreateImage(device, &imageInfo, nullptr, &ColorImage);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to create color image: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    VkMemoryRequirements requirements = {};
    vkGetImageMemoryRequirements(device, ColorImage, &requirements);
    u32 memoryType = 0;
    if (!Context.FindMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                memoryType))
    {
        Log(LogLevel::Error, "Vulkan: no device-local memory for color image\n");
        return false;
    }

    VkMemoryAllocateInfo allocationInfo = {};
    allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocationInfo.allocationSize = requirements.size;
    allocationInfo.memoryTypeIndex = memoryType;
    result = vkAllocateMemory(device, &allocationInfo, nullptr, &ColorMemory);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to allocate color memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }

    result = vkBindImageMemory(device, ColorImage, ColorMemory, 0);
    if (result != VK_SUCCESS)
    {
        Log(LogLevel::Error, "Vulkan: failed to bind color memory: %s (%d)\n",
            Vulkan::ResultName(result), result);
        return false;
    }
    return true;
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
    u8 xOffset = (GPU3D.RenderClearAttr2 >> 16) & 0xFF;
    u8 yOffset = (GPU3D.RenderClearAttr2 >> 24) & 0xFF;
    const u32 polygonId = GPU3D.RenderClearAttr1 & 0x3F000000;

    for (u32 y = 0; y < ScreenHeight; y++)
    {
        for (u32 x = 0; x < ScreenWidth; x++)
        {
            const u16 colorValue = GPU.ReadVRAMFlat_Texture<u16>(
                0x40000 + (static_cast<u32>(yOffset) << 9) + (static_cast<u32>(xOffset) << 1));
            const u16 depthValue = GPU.ReadVRAMFlat_Texture<u16>(
                0x60000 + (static_cast<u32>(yOffset) << 9) + (static_cast<u32>(xOffset) << 1));

            u32 red = (colorValue << 1) & 0x3E;
            u32 green = (colorValue >> 4) & 0x3E;
            u32 blue = (colorValue >> 9) & 0x3E;
            if (red) red++;
            if (green) green++;
            if (blue) blue++;
            const u32 alpha = (colorValue & 0x8000) ? 0x1F : 0;
            Framebuffer[y * ScreenWidth + x] = red | (green << 8) | (blue << 16) |
                                                (alpha << 24);

            (void)depthValue;
            (void)polygonId;
            xOffset++;
        }
        yOffset++;
    }
}

bool VulkanRenderer3D::SubmitFrame(bool bitmapClear)
{
    VkDevice device = Context.GetDevice();
    VkResult result = vkResetCommandBuffer(CommandBuffer, 0);
    if (result != VK_SUCCESS)
        return false;

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(CommandBuffer, &beginInfo);
    if (result != VK_SUCCESS)
        return false;

    VkImageMemoryBarrier toTransferDestination = {};
    toTransferDestination.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferDestination.srcAccessMask = ColorLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 :
                                          VK_ACCESS_TRANSFER_READ_BIT;
    toTransferDestination.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransferDestination.oldLayout = ColorLayout;
    toTransferDestination.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDestination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDestination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDestination.image = ColorImage;
    toTransferDestination.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferDestination.subresourceRange.levelCount = 1;
    toTransferDestination.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(CommandBuffer,
        ColorLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT :
                                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &toTransferDestination);

    VkImageSubresourceRange colorRange = {};
    colorRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorRange.levelCount = 1;
    colorRange.layerCount = 1;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = {ScreenWidth, ScreenHeight, 1};

    if (bitmapClear)
    {
        VkMemoryBarrier hostWriteBarrier = {};
        hostWriteBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hostWriteBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hostWriteBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_HOST_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &hostWriteBarrier,
                             0, nullptr, 0, nullptr);
        vkCmdCopyBufferToImage(CommandBuffer, ReadbackBuffer, ColorImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    }
    else
    {
        u32 red = (GPU3D.RenderClearAttr1 << 1) & 0x3E;
        u32 green = (GPU3D.RenderClearAttr1 >> 4) & 0x3E;
        u32 blue = (GPU3D.RenderClearAttr1 >> 9) & 0x3E;
        if (red) red++;
        if (green) green++;
        if (blue) blue++;

        VkClearColorValue clearColor = {};
        clearColor.uint32[0] = red;
        clearColor.uint32[1] = green;
        clearColor.uint32[2] = blue;
        clearColor.uint32[3] = (GPU3D.RenderClearAttr1 >> 16) & 0x1F;
        vkCmdClearColorImage(CommandBuffer, ColorImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clearColor, 1, &colorRange);
    }

    VkImageMemoryBarrier toTransferSource = toTransferDestination;
    toTransferSource.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransferSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toTransferSource.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &toTransferSource);
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

void VulkanRenderer3D::DestroyResources()
{
    VkDevice device = Context.GetDevice();
    if (device == VK_NULL_HANDLE)
        return;

    vkDeviceWaitIdle(device);
    if (Framebuffer)
        vkUnmapMemory(device, ReadbackMemory);
    if (ReadbackBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, ReadbackBuffer, nullptr);
    if (ReadbackMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, ReadbackMemory, nullptr);
    if (ColorImage != VK_NULL_HANDLE)
        vkDestroyImage(device, ColorImage, nullptr);
    if (ColorMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, ColorMemory, nullptr);
    if (Fence != VK_NULL_HANDLE)
        vkDestroyFence(device, Fence, nullptr);
    if (CommandBuffer != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device, Context.GetCommandPool(), 1, &CommandBuffer);
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
