//! Minimal hand-written WebGPU C ABI declarations for wgpu-native.
//!
//! This file intentionally does not use @cImport.  Keep this subset ABI-compatible
//! with webgpu.h (native and emdawnwebgpu use the same C ABI).  In particular,
//! asynchronous work is advanced with wgpuInstanceProcessEvents; wait-any and
//! device-poll entry points are intentionally absent from this declaration set.

const std = @import("std");

pub const Handle = ?*anyopaque;
pub const WGPUInstance = Handle;
pub const WGPUAdapter = Handle;
pub const WGPUDevice = Handle;
pub const WGPUQueue = Handle;
pub const WGPUBuffer = Handle;
pub const WGPUShaderModule = Handle;
pub const WGPUComputePipeline = Handle;
pub const WGPUPipelineLayout = Handle;
pub const WGPUBindGroupLayout = Handle;
pub const WGPUBindGroup = Handle;
pub const WGPUCommandEncoder = Handle;
pub const WGPUComputePassEncoder = Handle;
pub const WGPUCommandBuffer = Handle;
pub const WGPUSurface = Handle;

pub const WGPU_STRLEN: usize = std.math.maxInt(usize);
pub const WGPU_WHOLE_SIZE: u64 = std.math.maxInt(u64);

pub const WGPUBool = u32;
pub const WGPUFlags = u64;
pub const WGPUBufferUsage = WGPUFlags;
pub const WGPUMapMode = WGPUFlags;
pub const WGPUShaderStage = WGPUFlags;

pub const WGPUBufferUsage_MapRead: WGPUBufferUsage = 0x0001;
pub const WGPUBufferUsage_CopySrc: WGPUBufferUsage = 0x0004;
pub const WGPUBufferUsage_CopyDst: WGPUBufferUsage = 0x0008;
pub const WGPUBufferUsage_Uniform: WGPUBufferUsage = 0x0040;
pub const WGPUBufferUsage_Storage: WGPUBufferUsage = 0x0080;

pub const WGPUMapMode_Read: WGPUMapMode = 0x0001;
pub const WGPUShaderStage_Compute: WGPUShaderStage = 0x0004;

pub const WGPUCallbackMode = u32;
pub const WGPUCallbackMode_AllowProcessEvents: WGPUCallbackMode = 0x00000002;

pub const WGPUSType = u32;
pub const WGPUSType_ShaderSourceWGSL: WGPUSType = 0x00000002;

pub const WGPUStatus = u32;
pub const WGPUStatus_Success: WGPUStatus = 0x00000001;
pub const WGPUStatus_Error: WGPUStatus = 0x00000002;

pub const WGPURequestAdapterStatus = u32;
pub const WGPURequestAdapterStatus_Success: WGPURequestAdapterStatus = 0x00000001;
pub const WGPURequestAdapterStatus_Error: WGPURequestAdapterStatus = 0x00000004;

pub const WGPURequestDeviceStatus = u32;
pub const WGPURequestDeviceStatus_Success: WGPURequestDeviceStatus = 0x00000001;
pub const WGPURequestDeviceStatus_Error: WGPURequestDeviceStatus = 0x00000003;

pub const WGPUMapAsyncStatus = u32;
pub const WGPUMapAsyncStatus_Success: WGPUMapAsyncStatus = 0x00000001;

pub const WGPUErrorFilter = u32;
pub const WGPUErrorFilter_Validation: WGPUErrorFilter = 0x00000001;

pub const WGPUErrorType = u32;
pub const WGPUErrorType_NoError: WGPUErrorType = 0x00000001;

pub const WGPUPopErrorScopeStatus = u32;
pub const WGPUPopErrorScopeStatus_Success: WGPUPopErrorScopeStatus = 0x00000001;

pub const WGPUBackendType = u32;
pub const WGPUBackendType_Undefined: WGPUBackendType = 0x00000000;
pub const WGPUBackendType_Vulkan: WGPUBackendType = 0x00000006;

pub const WGPUBufferBindingType = u32;
pub const WGPUBufferBindingType_BindingNotUsed: WGPUBufferBindingType = 0x00000000;
pub const WGPUBufferBindingType_Uniform: WGPUBufferBindingType = 0x00000002;
pub const WGPUBufferBindingType_Storage: WGPUBufferBindingType = 0x00000003;
pub const WGPUBufferBindingType_ReadOnlyStorage: WGPUBufferBindingType = 0x00000004;

pub const WGPUSamplerBindingType = u32;
pub const WGPUSamplerBindingType_BindingNotUsed: WGPUSamplerBindingType = 0x00000000;
pub const WGPUTextureSampleType = u32;
pub const WGPUTextureViewDimension = u32;
pub const WGPUStorageTextureAccess = u32;
pub const WGPUTextureFormat = u32;

pub const WGPUStringView = extern struct {
    data: ?[*]const u8,
    length: usize,
};

pub const WGPUChainedStruct = extern struct {
    next: ?*WGPUChainedStruct,
    sType: WGPUSType,
};

pub const WGPUFuture = extern struct {
    id: u64,
};

pub const WGPURequestAdapterCallback = *const fn (
    status: WGPURequestAdapterStatus,
    adapter: WGPUAdapter,
    message: WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void;

pub const WGPURequestDeviceCallback = *const fn (
    status: WGPURequestDeviceStatus,
    device: WGPUDevice,
    message: WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void;

pub const WGPUBufferMapCallback = *const fn (
    status: WGPUMapAsyncStatus,
    message: WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void;

pub const WGPUPopErrorScopeCallback = *const fn (
    status: WGPUPopErrorScopeStatus,
    error_type: WGPUErrorType,
    message: WGPUStringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void;

pub const WGPURequestAdapterCallbackInfo = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    mode: WGPUCallbackMode,
    callback: ?WGPURequestAdapterCallback,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
};

pub const WGPURequestDeviceCallbackInfo = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    mode: WGPUCallbackMode,
    callback: ?WGPURequestDeviceCallback,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
};

pub const WGPUBufferMapCallbackInfo = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    mode: WGPUCallbackMode,
    callback: ?WGPUBufferMapCallback,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
};

pub const WGPUPopErrorScopeCallbackInfo = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    mode: WGPUCallbackMode,
    callback: ?WGPUPopErrorScopeCallback,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
};

pub const WGPUAdapterInfo = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    vendor: WGPUStringView,
    architecture: WGPUStringView,
    device: WGPUStringView,
    description: WGPUStringView,
    backendType: WGPUBackendType,
    adapterType: u32,
    vendorID: u32,
    deviceID: u32,
    subgroupMinSize: u32,
    subgroupMaxSize: u32,
};

pub const WGPUBufferDescriptor = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    label: WGPUStringView,
    usage: WGPUBufferUsage,
    size: u64,
    mappedAtCreation: WGPUBool,
};

pub const WGPUShaderModuleDescriptor = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    label: WGPUStringView,
};

pub const WGPUShaderSourceWGSL = extern struct {
    chain: WGPUChainedStruct,
    code: WGPUStringView,
};

pub const WGPUConstantEntry = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    key: WGPUStringView,
    value: f64,
};

pub const WGPUComputeState = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    module: WGPUShaderModule,
    entryPoint: WGPUStringView,
    constantCount: usize,
    constants: ?[*]const WGPUConstantEntry,
};

pub const WGPUComputePipelineDescriptor = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    label: WGPUStringView,
    layout: WGPUPipelineLayout,
    compute: WGPUComputeState,
};

pub const WGPUBufferBindingLayout = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    type: WGPUBufferBindingType,
    hasDynamicOffset: WGPUBool,
    minBindingSize: u64,
};

pub const WGPUSamplerBindingLayout = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    type: WGPUSamplerBindingType,
};

pub const WGPUTextureBindingLayout = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    sampleType: WGPUTextureSampleType,
    viewDimension: WGPUTextureViewDimension,
    multisampled: WGPUBool,
};

pub const WGPUStorageTextureBindingLayout = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    access: WGPUStorageTextureAccess,
    format: WGPUTextureFormat,
    viewDimension: WGPUTextureViewDimension,
};

pub const WGPUBindGroupLayoutEntry = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    binding: u32,
    visibility: WGPUShaderStage,
    bindingArraySize: u32,
    buffer: WGPUBufferBindingLayout,
    sampler: WGPUSamplerBindingLayout,
    texture: WGPUTextureBindingLayout,
    storageTexture: WGPUStorageTextureBindingLayout,
};

pub const WGPUBindGroupLayoutDescriptor = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    label: WGPUStringView,
    entryCount: usize,
    entries: ?[*]const WGPUBindGroupLayoutEntry,
};

pub const WGPUBindGroupEntry = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    binding: u32,
    buffer: WGPUBuffer,
    offset: u64,
    size: u64,
    sampler: Handle,
    textureView: Handle,
};

pub const WGPUBindGroupDescriptor = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    label: WGPUStringView,
    layout: WGPUBindGroupLayout,
    entryCount: usize,
    entries: ?[*]const WGPUBindGroupEntry,
};

pub const WGPUPipelineLayoutDescriptor = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    label: WGPUStringView,
    bindGroupLayoutCount: usize,
    bindGroupLayouts: ?[*]const WGPUBindGroupLayout,
    immediateSize: u32,
};

pub const WGPUCommandEncoderDescriptor = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    label: WGPUStringView,
};

pub const WGPUComputePassDescriptor = extern struct {
    nextInChain: ?*WGPUChainedStruct,
    label: WGPUStringView,
    timestampWrites: ?*const anyopaque,
};

// Request descriptors are passed as null in this backend.  They are intentionally
// opaque here to keep the binding limited to the compute subset actually used.
pub extern fn wgpuCreateInstance(descriptor: ?*const anyopaque) WGPUInstance;
pub extern fn wgpuInstanceProcessEvents(instance: WGPUInstance) void;
pub extern fn wgpuInstanceRequestAdapter(
    instance: WGPUInstance,
    options: ?*const anyopaque,
    callbackInfo: WGPURequestAdapterCallbackInfo,
) WGPUFuture;
pub extern fn wgpuAdapterRequestDevice(
    adapter: WGPUAdapter,
    descriptor: ?*const anyopaque,
    callbackInfo: WGPURequestDeviceCallbackInfo,
) WGPUFuture;
pub extern fn wgpuAdapterGetInfo(adapter: WGPUAdapter, info: *WGPUAdapterInfo) WGPUStatus;
pub extern fn wgpuAdapterInfoFreeMembers(info: WGPUAdapterInfo) void;

pub extern fn wgpuDeviceGetQueue(device: WGPUDevice) WGPUQueue;
pub extern fn wgpuDevicePushErrorScope(device: WGPUDevice, filter: WGPUErrorFilter) void;
pub extern fn wgpuDevicePopErrorScope(
    device: WGPUDevice,
    callbackInfo: WGPUPopErrorScopeCallbackInfo,
) WGPUFuture;
pub extern fn wgpuDeviceCreateShaderModule(
    device: WGPUDevice,
    descriptor: *const WGPUShaderModuleDescriptor,
) WGPUShaderModule;
pub extern fn wgpuDeviceCreateComputePipeline(
    device: WGPUDevice,
    descriptor: *const WGPUComputePipelineDescriptor,
) WGPUComputePipeline;
pub extern fn wgpuComputePipelineGetBindGroupLayout(
    pipeline: WGPUComputePipeline,
    groupIndex: u32,
) WGPUBindGroupLayout;
pub extern fn wgpuDeviceCreateBindGroupLayout(
    device: WGPUDevice,
    descriptor: *const WGPUBindGroupLayoutDescriptor,
) WGPUBindGroupLayout;
pub extern fn wgpuDeviceCreateBindGroup(
    device: WGPUDevice,
    descriptor: *const WGPUBindGroupDescriptor,
) WGPUBindGroup;
pub extern fn wgpuDeviceCreatePipelineLayout(
    device: WGPUDevice,
    descriptor: *const WGPUPipelineLayoutDescriptor,
) WGPUPipelineLayout;
pub extern fn wgpuDeviceCreateCommandEncoder(
    device: WGPUDevice,
    descriptor: ?*const WGPUCommandEncoderDescriptor,
) WGPUCommandEncoder;
pub extern fn wgpuCommandEncoderBeginComputePass(
    commandEncoder: WGPUCommandEncoder,
    descriptor: ?*const WGPUComputePassDescriptor,
) WGPUComputePassEncoder;
pub extern fn wgpuCommandEncoderFinish(
    commandEncoder: WGPUCommandEncoder,
    descriptor: ?*const anyopaque,
) WGPUCommandBuffer;
pub extern fn wgpuCommandEncoderCopyBufferToBuffer(
    commandEncoder: WGPUCommandEncoder,
    source: WGPUBuffer,
    sourceOffset: u64,
    destination: WGPUBuffer,
    destinationOffset: u64,
    size: u64,
) void;
pub extern fn wgpuComputePassEncoderSetPipeline(
    computePassEncoder: WGPUComputePassEncoder,
    pipeline: WGPUComputePipeline,
) void;
pub extern fn wgpuComputePassEncoderSetBindGroup(
    computePassEncoder: WGPUComputePassEncoder,
    groupIndex: u32,
    group: WGPUBindGroup,
    dynamicOffsetCount: usize,
    dynamicOffsets: ?[*]const u32,
) void;
pub extern fn wgpuComputePassEncoderDispatchWorkgroups(
    computePassEncoder: WGPUComputePassEncoder,
    workgroupCountX: u32,
    workgroupCountY: u32,
    workgroupCountZ: u32,
) void;
pub extern fn wgpuComputePassEncoderEnd(computePassEncoder: WGPUComputePassEncoder) void;

pub extern fn wgpuDeviceCreateBuffer(
    device: WGPUDevice,
    descriptor: *const WGPUBufferDescriptor,
) WGPUBuffer;
pub extern fn wgpuQueueWriteBuffer(
    queue: WGPUQueue,
    buffer: WGPUBuffer,
    bufferOffset: u64,
    data: ?*const anyopaque,
    size: usize,
) void;
pub extern fn wgpuQueueSubmit(
    queue: WGPUQueue,
    commandCount: usize,
    commands: ?[*]const WGPUCommandBuffer,
) void;
pub extern fn wgpuBufferMapAsync(
    buffer: WGPUBuffer,
    mode: WGPUMapMode,
    offset: usize,
    size: usize,
    callbackInfo: WGPUBufferMapCallbackInfo,
) WGPUFuture;
pub extern fn wgpuBufferGetMappedRange(
    buffer: WGPUBuffer,
    offset: usize,
    size: usize,
) ?*anyopaque;
// Browser WebGPU exposes read-only mappings through the const entry point;
// wgpu-native provides the same webgpu.h symbol.  Keep both in this shared ABI
// rather than maintaining a wasm-only binding fork.
pub extern fn wgpuBufferGetConstMappedRange(
    buffer: WGPUBuffer,
    offset: usize,
    size: usize,
) ?*const anyopaque;
pub extern fn wgpuBufferUnmap(buffer: WGPUBuffer) void;

pub extern fn wgpuInstanceRelease(instance: WGPUInstance) void;
pub extern fn wgpuAdapterRelease(adapter: WGPUAdapter) void;
pub extern fn wgpuDeviceRelease(device: WGPUDevice) void;
pub extern fn wgpuQueueRelease(queue: WGPUQueue) void;
pub extern fn wgpuBufferRelease(buffer: WGPUBuffer) void;
pub extern fn wgpuShaderModuleRelease(shaderModule: WGPUShaderModule) void;
pub extern fn wgpuComputePipelineRelease(pipeline: WGPUComputePipeline) void;
pub extern fn wgpuPipelineLayoutRelease(layout: WGPUPipelineLayout) void;
pub extern fn wgpuBindGroupLayoutRelease(layout: WGPUBindGroupLayout) void;
pub extern fn wgpuBindGroupRelease(bindGroup: WGPUBindGroup) void;
pub extern fn wgpuCommandEncoderRelease(commandEncoder: WGPUCommandEncoder) void;
pub extern fn wgpuComputePassEncoderRelease(computePassEncoder: WGPUComputePassEncoder) void;
pub extern fn wgpuCommandBufferRelease(commandBuffer: WGPUCommandBuffer) void;
