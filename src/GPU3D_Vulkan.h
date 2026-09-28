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

#ifndef GPU3D_VULKAN_H
#define GPU3D_VULKAN_H

#include <array>

#include "GPU3D.h"
#include "VulkanSupport.h"

namespace melonDS
{

class VulkanRenderer3D : public Renderer3D
{
public:
    VulkanRenderer3D(melonDS::GPU3D& gpu3D, Vulkan::Context& context) noexcept;
    ~VulkanRenderer3D() override;

    bool Init() override;
    void Reset() override;
    void SetRenderSettings(int scale) noexcept;

    void RenderFrame() override;
    u32* GetLine(int line) override;

private:
    static constexpr u32 ScreenWidth = 256;
    static constexpr u32 ScreenHeight = 192;
    static constexpr VkDeviceSize FramebufferSize = ScreenWidth * ScreenHeight * sizeof(u32);

    Vulkan::Context& Context;

    VkCommandBuffer CommandBuffer = VK_NULL_HANDLE;
    VkFence Fence = VK_NULL_HANDLE;
    VkImage ColorImage = VK_NULL_HANDLE;
    VkDeviceMemory ColorMemory = VK_NULL_HANDLE;
    VkBuffer ReadbackBuffer = VK_NULL_HANDLE;
    VkDeviceMemory ReadbackMemory = VK_NULL_HANDLE;
    u32* Framebuffer = nullptr;

    VkImageLayout ColorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::array<u32, ScreenWidth> ScrolledLine {};

    bool CreateColorImage();
    bool CreateReadbackBuffer();
    bool SubmitFrame(bool bitmapClear);
    void PrepareBitmapClear();
    void DestroyResources();
};

}

#endif // GPU3D_VULKAN_H
