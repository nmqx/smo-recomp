/* nvn_vk_interp.c - frame interpolation: between two of the game's frames a third is made from the motion the GPU's
 * optical flow engine finds between them (VK_NV_optical_flow, NVIDIA RTX 30 and later) and shown halfway, so the window
 * gets 120 frames a second while the game runs at its 60. Included by nvn_vk_draw.c (above vk_present2).
 *
 * graphics.ini frame_interpolation = optical_flow; SMO_VK_INTERP=1 / 0 wins. Off when the GPU has no VK_NV_optical_flow,
 * no queue family with optical flow and transfer, or no push descriptors (said in the log).
 *
 * A present of the game (vk_present2; not a loading screen's progress present, not the amiibo menu):
 *   1. render thread, graphics queue: the window texture (its crop) is copied 1:1 into col[k] (B8G8R8A8_UNORM, a ring of
 *      OFX_N), after the frame's own commands, in the same submission
 *   2. optical flow queue: the flow from col[k] (input) to col[p] (reference: the frame before) and back, on a 4x4 grid
 *      in S10.5 fixed point, copied into buf[k] (forward, then backward). session[k] has those images bound once.
 *   3. graphics queue: the compute stage below makes mid[k] (R8G8B8A8_UNORM): for each pixel, the pair of samples of the
 *      two frames that the forward flow, the backward flow or no motion at all join best (smallest colour difference),
 *      averaged. Still parts (the 2D interface) keep the no-motion pair.
 *   4. the presenter thread shows mid[k] as soon as the GPU has made it, then col[k] half a game frame later.
 * The game's own frame is thus shown about half a frame later than without interpolation.
 * When the game's next frame is late (loading in the background, a first use of a shader), the presenter does not wait:
 * it shows the newest frame moved on along its own motion (the extrapolation stage below, the same idea as the
 * asynchronous reprojection of VR headsets), half a game frame further each time, at most SMO_VK_INTERP_EXT_MAX (6) times,
 * so a stall of up to about 50 ms still gives 120 different pictures a second. SMO_VK_INTERP_EXT=0 turns that off.
 *
 * The presenter thread presents only while the render thread leaves the swapchain alone: every other present of the
 * render thread (and every swapchain change) first waits until the presenter has nothing left (ofx_drain); the graphics
 * queue is shared under g_qlock. A swapchain the presenter finds out of date is made again by the render thread at its
 * next present (O.swap_bad).
 *
 * SMO_VK_INTERP_STATS=1: every 600 made frames, which pair their pixels took (forward / backward / no motion).
 * SMO_VK_INTERP_SHOT=<play frame>[,...]: pictures of a made frame and its two neighbours (ofx_shot). */

#define OFX_N 4                                 /* the ring of slots: a frame and its flow stay readable for the presenter while the next ones are made */
static PFN_vkCreateOpticalFlowSessionNV vkCreateOpticalFlowSessionNV;
static PFN_vkDestroyOpticalFlowSessionNV vkDestroyOpticalFlowSessionNV;
static PFN_vkBindOpticalFlowSessionImageNV vkBindOpticalFlowSessionImageNV;
static PFN_vkCmdOpticalFlowExecuteNV vkCmdOpticalFlowExecuteNV;
static PFN_vkCmdPipelineBarrier2 vkCmdPipelineBarrier2;
static PFN_vkWaitSemaphores vkWaitSemaphores;

typedef struct { VkImage img; uint64_t wait; int half, slot; } OfxItem;
static struct {
    int want, on, failed;
    uint32_t family;                                /* the queue family with optical flow (and transfer) */
    VkDeviceQueueCreateInfo qi[2];
    float prio, gprio[2];
    VkQueue pqueue;                                 /* the presenter's: a second queue of the graphics family when there is one, else V.queue */
    VkPhysicalDeviceOpticalFlowFeaturesNV feat;
    VkQueue queue;
    int ready, w, h, gw, gh, flow_init;
    VkImage col[OFX_N], mid[OFX_N], flow[2];
    VMem col_mem[OFX_N], mid_mem[OFX_N], flow_mem[2], buf_mem[OFX_N];
    VkImageView col_view[OFX_N], mid_view[OFX_N], flow_view[2];
    VkBuffer buf[OFX_N];
    VkOpticalFlowSessionNV session[OFX_N];          /* session[k]: input col[k], reference col[k - 1] */
    VkCommandPool of_pool;
    VkCommandBuffer of_cmd[OFX_N];
    VkFence of_fence[OFX_N];
    int of_pending[OFX_N];
    VkSemaphore g2of[OFX_N], of2g[OFX_N];
    VkSemaphore tl;                                 /* render side: the value at which a slot's images are made */
    uint64_t tl_val;
    VkSampler smp;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkPipeline pipe;
    VkBuffer stats; VMem stats_mem; int stats_on;
    int k, have_prev;
    LONG64 t_last;
    double period;                                  /* game frame interval, ms (smoothed) */
    /* the presenter */
    HANDLE thread;
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE cv, idle_cv;
    OfxItem q[8];
    int qh, qn, busy, stop;
    int slot_items[OFX_N];                          /* items of a slot not yet submitted by the presenter */
    VkSemaphore ptl;                                /* presenter: signalled by each of its blits */
    uint64_t pval, p_used[OFX_N];
    VkCommandPool p_pool;
    VkCommandBuffer p_cmd[4];
    VkFence p_fence[4];
    int p_pending[4], p_i, p_ai;
    VkSemaphore p_acq[8], p_rend[16];
    volatile LONG swap_bad;
    volatile LONG64 made, shown, skipped, late;
    /* extrapolation: frames predicted from the newest frame of the game while its next one is late */
    int ext_on, ext_max, no_ext;                    /* no_ext: the render thread presents on its own (set by ofx_drain, cleared by ofx_push) */
    int slot_flow[OFX_N];                           /* the slot's flow was measured (it had a frame before it) */
    uint64_t slot_wait[OFX_N];
    VkImage ext[2]; VMem ext_mem[2]; VkImageView ext_view[2];
    VkDescriptorSetLayout xdsl;
    VkPipelineLayout xpl;
    VkPipeline xpipe;
    volatile LONG64 ext_shown, ext_runs, mid_dropped, ext_longest;
} O;

/* device creation (vkd_device_info, before the features are read): the extension, its feature and a queue of the optical
 * flow family next to the graphics queue */
static void ofx_device_info(VkDeviceCreateInfo *di, const VkExtensionProperties *dext, uint32_t ndext, void ***tail) {
    const char *e = getenv("SMO_VK_INTERP");
    O.want = e && e[0] ? atoi(e) != 0 : gfx_options()->interp;
    if (!O.want) return;
    if (!has_name(dext, ndext, VK_NV_OPTICAL_FLOW_EXTENSION_NAME)) { printf("nvn_vk_interp: frame_interpolation asked for, but the GPU has no VK_NV_optical_flow (NVIDIA RTX 30 and later): off\n"); O.want = 0; return; }
    VkQueueFamilyProperties q[32];
    uint32_t nq = 32;
    int fam = -1;
    vkGetPhysicalDeviceQueueFamilyProperties(V.gpu, &nq, q);
    for (uint32_t i = 0; i < nq && fam < 0; i++) if ((q[i].queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) && (q[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && i != V.family) fam = (int)i;
    if (fam < 0) { printf("nvn_vk_interp: no queue family with optical flow and transfer: frame interpolation off\n"); O.want = 0; return; }
    O.family = (uint32_t)fam;
    D.exts[D.next++] = VK_NV_OPTICAL_FLOW_EXTENSION_NAME;
    O.feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV;
    **tail = &O.feat; *tail = &O.feat.pNext;
    O.qi[0] = di->pQueueCreateInfos[0];
    if (q[V.family].queueCount >= 2) { O.qi[0].queueCount = 2; O.qi[0].pQueuePriorities = O.gprio; }      /* the presenter's own queue (index 1) */
    O.prio = 1.0f; O.gprio[0] = O.gprio[1] = 1.0f;
    O.qi[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; O.qi[1].queueFamilyIndex = O.family; O.qi[1].queueCount = 1; O.qi[1].pQueuePriorities = &O.prio;
    di->queueCreateInfoCount = 2; di->pQueueCreateInfos = O.qi;
}

static const char k_ofx_glsl[] =
    "#version 450\n"
    "layout(local_size_x = 16, local_size_y = 8) in;\n"
    "layout(set = 0, binding = 0) uniform sampler2D prevTex;\n"
    "layout(set = 0, binding = 1) uniform sampler2D curTex;\n"
    "layout(set = 0, binding = 2, rgba8) uniform writeonly image2D outImg;\n"
    "layout(set = 0, binding = 3, std430) readonly buffer Flow { int v[]; } flow;\n"     /* forward (cur -> prev), then backward (prev -> cur): x low 16 bits, y high, S10.5 */
    "layout(set = 0, binding = 4, std430) buffer Stats { uint n[4]; } stats;\n"
    "layout(push_constant) uniform Pc { ivec2 size; ivec2 grid; int stats; } pc;\n"
    "shared uint cnt[3];\n"
    "vec2 fetch(int base, ivec2 c) {\n"
    "    c = clamp(c, ivec2(0), pc.grid - 1);\n"
    "    int w = flow.v[base + c.y * pc.grid.x + c.x];\n"
    "    return vec2(float((w << 16) >> 16), float(w >> 16)) * (1.0 / 32.0);\n"
    "}\n"
    "vec2 flowAt(int base, vec2 p) {\n"                 /* bilinear between the centres of the 4x4 cells */
    "    vec2 g = p * 0.25 - 0.5;\n"
    "    ivec2 c = ivec2(floor(g));\n"
    "    vec2 f = g - vec2(c);\n"
    "    return mix(mix(fetch(base, c), fetch(base, c + ivec2(1, 0)), f.x), mix(fetch(base, c + ivec2(0, 1)), fetch(base, c + ivec2(1, 1)), f.x), f.y);\n"
    "}\n"
    "void main() {\n"
    "    if (gl_LocalInvocationIndex == 0u) { cnt[0] = 0u; cnt[1] = 0u; cnt[2] = 0u; }\n"
    "    barrier();\n"
    "    ivec2 q = ivec2(gl_GlobalInvocationID.xy);\n"
    "    if (q.x < pc.size.x && q.y < pc.size.y) {\n"
    "        vec2 p = vec2(q) + 0.5, inv = 1.0 / vec2(pc.size);\n"
    "        int n = pc.grid.x * pc.grid.y;\n"
    "        vec2 fw = flowAt(0, p), bw = flowAt(n, p);\n"
    "        vec3 a0 = texture(prevTex, (p + 0.5 * fw) * inv).rgb, a1 = texture(curTex, (p - 0.5 * fw) * inv).rgb;\n"
    "        vec3 b0 = texture(prevTex, (p - 0.5 * bw) * inv).rgb, b1 = texture(curTex, (p + 0.5 * bw) * inv).rgb;\n"
    "        vec3 c0 = texture(prevTex, p * inv).rgb, c1 = texture(curTex, p * inv).rgb;\n"
    "        float ea = dot(abs(a0 - a1), vec3(1.0)), eb = dot(abs(b0 - b1), vec3(1.0)), ec = dot(abs(c0 - c1), vec3(1.0)) - 0.02;\n"
    "        vec3 r = ea <= eb ? 0.5 * (a0 + a1) : 0.5 * (b0 + b1);\n"
    "        uint which = ea <= eb ? 0u : 1u;\n"
    "        if (ec <= min(ea, eb)) { r = 0.5 * (c0 + c1); which = 2u; }\n"
    "        imageStore(outImg, q, vec4(r, 1.0));\n"
    "        if (pc.stats != 0) atomicAdd(cnt[which], 1u);\n"
    "    }\n"
    "    barrier();\n"
    "    if (pc.stats != 0 && gl_LocalInvocationIndex == 0u) { atomicAdd(stats.n[0], cnt[0]); atomicAdd(stats.n[1], cnt[1]); atomicAdd(stats.n[2], cnt[2]); }\n"
    "}\n";

/* extrapolation: the newest frame moved on by alpha game frames along its own motion (the flow from it to the frame before
 * points back in time: cur(x) ~ prev(x + F(x)), so what will be at p was at p + alpha * F(p)) */
static const char k_ofx_ext_glsl[] =
    "#version 450\n"
    "layout(local_size_x = 16, local_size_y = 8) in;\n"
    "layout(set = 0, binding = 0) uniform sampler2D curTex;\n"
    "layout(set = 0, binding = 1, rgba8) uniform writeonly image2D outImg;\n"
    "layout(set = 0, binding = 2, std430) readonly buffer Flow { int v[]; } flow;\n"
    "layout(push_constant) uniform Pc { ivec2 size; ivec2 grid; float alpha; } pc;\n"
    "vec2 fetch(ivec2 c) {\n"
    "    c = clamp(c, ivec2(0), pc.grid - 1);\n"
    "    int w = flow.v[c.y * pc.grid.x + c.x];\n"
    "    return vec2(float((w << 16) >> 16), float(w >> 16)) * (1.0 / 32.0);\n"
    "}\n"
    "void main() {\n"
    "    ivec2 q = ivec2(gl_GlobalInvocationID.xy);\n"
    "    if (q.x >= pc.size.x || q.y >= pc.size.y) return;\n"
    "    vec2 p = vec2(q) + 0.5;\n"
    "    vec2 g = p * 0.25 - 0.5;\n"
    "    ivec2 c = ivec2(floor(g));\n"
    "    vec2 f = g - vec2(c);\n"
    "    vec2 fw = mix(mix(fetch(c), fetch(c + ivec2(1, 0)), f.x), mix(fetch(c + ivec2(0, 1)), fetch(c + ivec2(1, 1)), f.x), f.y);\n"
    "    vec2 d = pc.alpha * fw;\n"
    "    float l = length(d);\n"
    "    if (l > 96.0) d *= 96.0 / l;\n"                 /* a wrong vector far away would tear the picture */
    "    imageStore(outImg, q, vec4(texture(curTex, (p + d) / vec2(pc.size)).rgb, 1.0));\n"
    "}\n";
static VkPipeline ofx_compute(const char *glsl, VkPipelineLayout pl) {
    uint32_t *words = NULL;
    size_t count = 0;
    char log[2000];
    if (!g_glsl_to_spirv(5, glsl, &words, &count, log, sizeof log)) { printf("nvn_vk_interp: a stage does not compile: %s\n", log); return NULL; }
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    VkShaderModule mod;
    VkPipeline pipe;
    mi.codeSize = count * 4; mi.pCode = words;
    VKC(vkCreateShaderModule(V.device, &mi, NULL, &mod));
    g_spirv_free(words);
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = mod; ci.stage.pName = "main"; ci.layout = pl;
    VKC(vkCreateComputePipelines(V.device, NULL, 1, &ci, NULL, &pipe));
    vkDestroyShaderModule(V.device, mod, NULL);
    return pipe;
}
static void ofx_layout(const VkDescriptorType *types, int n, uint32_t pc_size, VkDescriptorSetLayout *dsl, VkPipelineLayout *pl) {
    VkDescriptorSetLayoutBinding b[8];
    memset(b, 0, sizeof b);
    for (int i = 0; i < n; i++) { b[i].binding = (uint32_t)i; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; b[i].descriptorType = types[i]; }
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR; li.bindingCount = (uint32_t)n; li.pBindings = b;
    VKC(vkCreateDescriptorSetLayout(V.device, &li, NULL, dsl));
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pr;
    VKC(vkCreatePipelineLayout(V.device, &pli, NULL, pl));
}
static DWORD WINAPI ofx_thread(void *arg);
/* after the device exists (vkd_init): functions, queue, command pools, sync objects, the compute stage, the presenter */
static void ofx_start(void) {
    if (!O.want) return;
    if (!O.feat.opticalFlow) { printf("nvn_vk_interp: the device does not offer the opticalFlow feature: frame interpolation off\n"); return; }
    if (!D.has_push) { printf("nvn_vk_interp: no push descriptors (VK_KHR_push_descriptor): frame interpolation off\n"); return; }
#define G(n) if (!(n = (PFN_##n)vkGetDeviceProcAddr(V.device, #n))) { printf("nvn_vk_interp: the device has no %s: frame interpolation off\n", #n); return; }
    G(vkCreateOpticalFlowSessionNV) G(vkDestroyOpticalFlowSessionNV) G(vkBindOpticalFlowSessionImageNV) G(vkCmdOpticalFlowExecuteNV) G(vkCmdPipelineBarrier2) G(vkWaitSemaphores)
#undef G
    vkGetDeviceQueue(V.device, O.family, 0, &O.queue);
    if (O.qi[0].queueCount >= 2) vkGetDeviceQueue(V.device, V.family, 1, &O.pqueue); else O.pqueue = V.queue;
    VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphoreTypeCreateInfo st = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    VkSemaphoreCreateInfo ti = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pi.queueFamilyIndex = O.family;
    VKC(vkCreateCommandPool(V.device, &pi, NULL, &O.of_pool));
    pi.queueFamilyIndex = V.family;
    VKC(vkCreateCommandPool(V.device, &pi, NULL, &O.p_pool));
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = OFX_N; ai.commandPool = O.of_pool;
    VKC(vkAllocateCommandBuffers(V.device, &ai, O.of_cmd));
    ai.commandBufferCount = 4; ai.commandPool = O.p_pool;
    VKC(vkAllocateCommandBuffers(V.device, &ai, O.p_cmd));
    for (int i = 0; i < OFX_N; i++) { VKC(vkCreateFence(V.device, &fi, NULL, &O.of_fence[i])); VKC(vkCreateSemaphore(V.device, &si, NULL, &O.g2of[i])); VKC(vkCreateSemaphore(V.device, &si, NULL, &O.of2g[i])); }
    for (int i = 0; i < 4; i++) VKC(vkCreateFence(V.device, &fi, NULL, &O.p_fence[i]));
    for (int i = 0; i < 8; i++) VKC(vkCreateSemaphore(V.device, &si, NULL, &O.p_acq[i]));
    for (int i = 0; i < 16; i++) VKC(vkCreateSemaphore(V.device, &si, NULL, &O.p_rend[i]));
    st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE; ti.pNext = &st;
    VKC(vkCreateSemaphore(V.device, &ti, NULL, &O.tl));
    VKC(vkCreateSemaphore(V.device, &ti, NULL, &O.ptl));
    {   VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = ci.minFilter = VK_FILTER_LINEAR; ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VKC(vkCreateSampler(V.device, &ci, NULL, &O.smp));
    }
    {   static const VkDescriptorType mid_types[5] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                      ext_types[3] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        ofx_layout(mid_types, 5, 20, &O.dsl, &O.pl);
        ofx_layout(ext_types, 3, 20, &O.xdsl, &O.xpl);
        if (!(O.pipe = ofx_compute(k_ofx_glsl, O.pl)) || !(O.xpipe = ofx_compute(k_ofx_ext_glsl, O.xpl))) return;
    }
    {   const char *e = getenv("SMO_VK_INTERP_EXT"), *m = getenv("SMO_VK_INTERP_EXT_MAX");
        O.ext_on = !(e && e[0] && atoi(e) == 0);
        O.ext_max = m && atoi(m) > 0 ? atoi(m) : 6;
    }
    O.stats = vbuf_new(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &O.stats_mem);
    memset(O.stats_mem.map, 0, 16);
    O.stats_on = getenv("SMO_VK_INTERP_STATS") && atoi(getenv("SMO_VK_INTERP_STATS"));
    O.period = 1000.0 / 60.0;
    InitializeCriticalSection(&O.lock); InitializeConditionVariable(&O.cv); InitializeConditionVariable(&O.idle_cv);
    O.thread = CreateThread(NULL, 0, ofx_thread, NULL, 0, NULL);
    SetThreadPriority(O.thread, THREAD_PRIORITY_HIGHEST);
    O.on = 1;
    printf("nvn_vk_interp: ON: a frame made by the optical flow engine (queue family %u) between two of the game's, shown halfway: 120 frames a second; "
           "a late frame of the game is bridged by up to %d frames predicted from the one before (%s; SMO_VK_INTERP_EXT=0 / SMO_VK_INTERP_EXT_MAX=<n>)\n",
           (unsigned)O.family, O.ext_max, O.ext_on ? "on" : "OFF");
}

static void ofx_drain(void) {                       /* render thread: until the presenter has shown everything it was given */
    if (!O.thread) return;
    EnterCriticalSection(&O.lock);
    O.no_ext = 1;                                   /* and predicts nothing until the next frame it is given */
    while (O.qn || O.busy) SleepConditionVariableCS(&O.idle_cv, &O.lock, 100);
    LeaveCriticalSection(&O.lock);
}
static void ofx_free(void) {
    if (!O.ready) return;
    ofx_drain();
    cmd_wait_all();
    for (int i = 0; i < OFX_N; i++) if (O.of_pending[i]) { vkWaitForFences(V.device, 1, &O.of_fence[i], VK_TRUE, 5000000000ull); vkResetFences(V.device, 1, &O.of_fence[i]); O.of_pending[i] = 0; }
    vkDeviceWaitIdle(V.device);
    for (int i = 0; i < OFX_N; i++) {
        vkDestroyOpticalFlowSessionNV(V.device, O.session[i], NULL);
        vkDestroyImageView(V.device, O.col_view[i], NULL); vkDestroyImage(V.device, O.col[i], NULL); vmem_free(O.col_mem[i]);
        vkDestroyImageView(V.device, O.mid_view[i], NULL); vkDestroyImage(V.device, O.mid[i], NULL); vmem_free(O.mid_mem[i]);
        vkDestroyBuffer(V.device, O.buf[i], NULL); vmem_free(O.buf_mem[i]);
    }
    for (int i = 0; i < 2; i++) { vkDestroyImageView(V.device, O.flow_view[i], NULL); vkDestroyImage(V.device, O.flow[i], NULL); vmem_free(O.flow_mem[i]); }
    for (int i = 0; i < 2; i++) { vkDestroyImageView(V.device, O.ext_view[i], NULL); vkDestroyImage(V.device, O.ext[i], NULL); vmem_free(O.ext_mem[i]); }
    O.ready = 0; O.have_prev = 0;
}
static VkImage ofx_image(int w, int h, VkFormat f, VkFlags usage, int concurrent, VkOpticalFlowUsageFlagsNV of_usage, VMem *mem, VkImageView *view) {
    uint32_t fams[2] = {V.family, O.family};
    VkOpticalFlowImageFormatInfoNV oi = {VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV, NULL, of_usage};
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    VkMemoryRequirements rq;
    VkImage img;
    ci.pNext = of_usage ? &oi : NULL;
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = f; ci.extent.width = (uint32_t)w; ci.extent.height = (uint32_t)h; ci.extent.depth = 1;
    ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = usage;
    ci.sharingMode = concurrent ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE; ci.queueFamilyIndexCount = concurrent ? 2 : 0; ci.pQueueFamilyIndices = fams;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKC(vkCreateImage(V.device, &ci, NULL, &img));
    vkGetImageMemoryRequirements(V.device, img, &rq);
    *mem = vmem_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    VKC(vkBindImageMemory(V.device, img, mem->mem, mem->off));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = f; vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
    VKC(vkCreateImageView(V.device, &vi, NULL, view));
    return img;
}
/* the images, buffers and sessions for frames of w x h; the colour images and the made frames go to GENERAL (render cmd) */
static int ofx_make(int w, int h) {
    O.w = w; O.h = h; O.gw = (w + 3) / 4; O.gh = (h + 3) / 4;
    for (int i = 0; i < OFX_N; i++) {
        O.col[i] = ofx_image(w, h, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 1, VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV, &O.col_mem[i], &O.col_view[i]);
        O.mid[i] = ofx_image(w, h, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, 0, &O.mid_mem[i], &O.mid_view[i]);
        uint32_t fams[2] = {V.family, O.family};
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        VkMemoryRequirements rq;
        bi.size = (uint64_t)O.gw * O.gh * 8; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_CONCURRENT; bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fams;
        VKC(vkCreateBuffer(V.device, &bi, NULL, &O.buf[i]));
        vkGetBufferMemoryRequirements(V.device, O.buf[i], &rq);
        O.buf_mem[i] = vmem_alloc(rq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 7);       /* class 7: blocks of their own (buffers next to optimal images need the granularity) */
        VKC(vkBindBufferMemory(V.device, O.buf[i], O.buf_mem[i].mem, O.buf_mem[i].off));
    }
    for (int i = 0; i < 2; i++) O.ext[i] = ofx_image(w, h, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, 0, &O.ext_mem[i], &O.ext_view[i]);
    for (int i = 0; i < 2; i++) O.flow[i] = ofx_image(O.gw, O.gh, VK_FORMAT_R16G16_SFIXED5_NV, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV, &O.flow_mem[i], &O.flow_view[i]);
    for (int i = 0; i < OFX_N; i++) {
        VkOpticalFlowSessionCreateInfoNV ci = {VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV};
        ci.width = (uint32_t)w; ci.height = (uint32_t)h; ci.imageFormat = VK_FORMAT_B8G8R8A8_UNORM; ci.flowVectorFormat = VK_FORMAT_R16G16_SFIXED5_NV;
        ci.outputGridSize = VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV; ci.performanceLevel = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_NV; ci.flags = VK_OPTICAL_FLOW_SESSION_CREATE_BOTH_DIRECTIONS_BIT_NV;
        VkResult r = vkCreateOpticalFlowSessionNV(V.device, &ci, NULL, &O.session[i]);
        if (r != VK_SUCCESS) { printf("nvn_vk_interp: vkCreateOpticalFlowSessionNV %dx%d: %s: frame interpolation off\n", w, h, vk_result_name(r)); return 0; }
        int p = (i + OFX_N - 1) % OFX_N;
        VKC(vkBindOpticalFlowSessionImageNV(V.device, O.session[i], VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV, O.col_view[i], VK_IMAGE_LAYOUT_GENERAL));
        VKC(vkBindOpticalFlowSessionImageNV(V.device, O.session[i], VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV, O.col_view[p], VK_IMAGE_LAYOUT_GENERAL));
        VKC(vkBindOpticalFlowSessionImageNV(V.device, O.session[i], VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV, O.flow_view[0], VK_IMAGE_LAYOUT_GENERAL));
        VKC(vkBindOpticalFlowSessionImageNV(V.device, O.session[i], VK_OPTICAL_FLOW_SESSION_BINDING_POINT_BACKWARD_FLOW_VECTOR_NV, O.flow_view[1], VK_IMAGE_LAYOUT_GENERAL));
    }
    pass_end();
    cmd_begin();
    for (int i = 0; i < OFX_N; i++) { image_barrier(O.col[i], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL); image_barrier(O.mid[i], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL); O.slot_flow[i] = 0; }
    for (int i = 0; i < 2; i++) image_barrier(O.ext[i], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    O.flow_init = 0; O.ready = 1; O.have_prev = 0; O.k = 0;
    printf("nvn_vk_interp: frames of %dx%d, flow on a %dx%d grid (4x4 pixels a vector, both directions)\n", w, h, O.gw, O.gh);
    return 1;
}

static void ofx_push(VkImage img, uint64_t wait, int half, int slot) {
    EnterCriticalSection(&O.lock);
    while (O.qn == 8) SleepConditionVariableCS(&O.idle_cv, &O.lock, 100);
    O.q[(O.qh + O.qn) % 8] = (OfxItem){img, wait, half, slot};
    O.qn++; O.slot_items[slot]++;
    O.no_ext = 0;
    WakeAllConditionVariable(&O.cv);
    LeaveCriticalSection(&O.lock);
}
/* render thread: slot k is no longer read by the presenter or the optical flow queue (timed: the render thread's wait) */
static void ofx_wait_slot2(int k);
static LONG64 g_ofx_wait_tk, g_ofx_wait_max;
static void ofx_wait_slot(int k) {
    LONG64 t0 = tk();
    ofx_wait_slot2(k);
    LONG64 d = tk() - t0;
    g_ofx_wait_tk += d;
    if (d > g_ofx_wait_max) g_ofx_wait_max = d;
}
static void ofx_wait_slot2(int k) {
    EnterCriticalSection(&O.lock);
    while (O.slot_items[k]) SleepConditionVariableCS(&O.idle_cv, &O.lock, 100);
    uint64_t v = O.p_used[k];
    LeaveCriticalSection(&O.lock);
    if (v) { VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO}; wi.semaphoreCount = 1; wi.pSemaphores = &O.ptl; wi.pValues = &v; vkWaitSemaphores(V.device, &wi, 2000000000ull); }
    for (int j = 0; j < 2; j++) {                   /* the flow runs that read col[k]: its own (input) and the next slot's (reference); long finished */
        int i = (k + j) % OFX_N;
        if (O.of_pending[i]) { VKC(vkWaitForFences(V.device, 1, &O.of_fence[i], VK_TRUE, 5000000000ull)); VKC(vkResetFences(V.device, 1, &O.of_fence[i])); O.of_pending[i] = 0; }
    }
}
static void ofx_barrier2(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    b.srcStageMask = ss; b.srcAccessMask = sa; b.dstStageMask = ds; b.dstAccessMask = da; b.oldLayout = from; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; b.subresourceRange.levelCount = 1; b.subresourceRange.layerCount = 1;
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}
/* the optical flow queue's part of slot k: flow both ways, copied into buf[k]; waits for g2of[k], signals of2g[k] */
static void ofx_flow(int k) {
    VkCommandBuffer cb = O.of_cmd[k];
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkResetCommandBuffer(cb, 0));
    VKC(vkBeginCommandBuffer(cb, &bi));
    for (int i = 0; i < 2; i++)                     /* the copy of the frame before has read the flow images */
        ofx_barrier2(cb, O.flow[i], O.flow_init ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV, VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV);
    O.flow_init = 1;
    VkOpticalFlowExecuteInfoNV ei = {VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV};
    ei.flags = VK_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_NV;      /* the session's last run was OFX_N frames ago */
    vkCmdOpticalFlowExecuteNV(cb, O.session[k], &ei);
    for (int i = 0; i < 2; i++) ofx_barrier2(cb, O.flow[i], VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV, VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < 2; i++) {
        VkBufferImageCopy c;
        memset(&c, 0, sizeof c);
        c.bufferOffset = (uint64_t)i * O.gw * O.gh * 4;
        c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; c.imageSubresource.layerCount = 1;
        c.imageExtent.width = (uint32_t)O.gw; c.imageExtent.height = (uint32_t)O.gh; c.imageExtent.depth = 1;
        vkCmdCopyImageToBuffer(cb, O.flow[i], VK_IMAGE_LAYOUT_GENERAL, O.buf[k], 1, &c);
    }
    VKC(vkEndCommandBuffer(cb));
    VkFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo sb = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    sb.waitSemaphoreCount = 1; sb.pWaitSemaphores = &O.g2of[k]; sb.pWaitDstStageMask = &stage;
    sb.commandBufferCount = 1; sb.pCommandBuffers = &cb;
    sb.signalSemaphoreCount = 1; sb.pSignalSemaphores = &O.of2g[k];
    VKC(vkQueueSubmit(O.queue, 1, &sb, O.of_fence[k]));
    O.of_pending[k] = 1;
}

/* SMO_VK_INTERP_SHOT=<n>[,<n>...]: the first frame made at or after the n-th play frame, with the two it is made from, as
 * interp_<m>_0_before.ppm, _1_made, _2_after in SMO_SHOT_DIR (render thread, right after its submission: waits for the GPU) */
static void ofx_shot(int p, int k, uint64_t v) {
    VMem m;
    size_t one = (size_t)O.w * O.h * 4;
    VkBuffer b = vbuf_new(one * 3, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &m);
    VkImage img[3] = {O.col[p], O.mid[k], O.col[k]};
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1; wi.pSemaphores = &O.tl; wi.pValues = &v;
    vkWaitSemaphores(V.device, &wi, 5000000000ull);
    cmd_begin();
    for (int i = 0; i < 3; i++) {
        VkBufferImageCopy c;
        memset(&c, 0, sizeof c);
        c.bufferOffset = one * (size_t)i; c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; c.imageSubresource.layerCount = 1;
        c.imageExtent.width = (uint32_t)O.w; c.imageExtent.height = (uint32_t)O.h; c.imageExtent.depth = 1;
        vkCmdCopyImageToBuffer(R.cmd, img[i], VK_IMAGE_LAYOUT_GENERAL, b, 1, &c);
    }
    cmd_wait_all();
    static const char *const k_name[3] = {"0_before", "1_made", "2_after"};
    const char *dir = getenv("SMO_SHOT_DIR");
    uint8_t *row = malloc((size_t)O.w * 3);
    for (int i = 0; i < 3; i++) {
        char path[300];
        snprintf(path, sizeof path, "%s%sinterp_%lld_%s.ppm", dir ? dir : "", dir ? "/" : "", (long long)O.made, k_name[i]);
        FILE *f = fopen(path, "wb");
        if (!f) continue;
        fprintf(f, "P6\n%d %d\n255\n", O.w, O.h);
        const uint8_t *px = (const uint8_t *)m.map + one * (size_t)i;
        int bgra = i != 1;                          /* col images are B8G8R8A8, the made frame R8G8B8A8 */
        for (int y = 0; y < O.h; y++) {
            for (int x = 0; x < O.w; x++) { const uint8_t *s = px + ((size_t)y * O.w + x) * 4; row[x * 3] = s[bgra ? 2 : 0]; row[x * 3 + 1] = s[1]; row[x * 3 + 2] = s[bgra ? 0 : 2]; }
            fwrite(row, 1, (size_t)O.w * 3, f);
        }
        fclose(f);
        printf("nvn_vk_interp: picture %s\n", path);
    }
    free(row);
    vkDestroyBuffer(V.device, b, NULL);
    vmem_free(m);
    fflush(stdout);
}

/* vk_present2, render thread: the frame of vt (its crop s4) goes to the presenter, with the frame made between it and the
 * one before. 1 = taken (nothing else to present), 0 = present it the usual way. */
static int ofx_present(VTex *vt, const int *s4) {
    int w = s4[2], h = s4[3];
    if (!O.on || O.failed || w < 64 || h < 64) return 0;
    if (!O.ready || O.w != w || O.h != h) { ofx_free(); if (!ofx_make(w, h)) { O.failed = 1; return 0; } }
    {   LONG64 now = tk();
        LARGE_INTEGER fq; QueryPerformanceFrequency(&fq);
        double ms = O.t_last ? (double)(now - O.t_last) * 1000.0 / (double)fq.QuadPart : 0.0;
        if (ms > 8.0 && ms < 40.0) O.period += (ms - O.period) * 0.1;
        if (ms > 40.0) O.have_prev = 0;             /* a long gap (loading, a hold): the frame before is not its neighbour */
        O.t_last = now;
    }
    int k = O.k, p = (k + OFX_N - 1) % OFX_N;
    ofx_wait_slot(k);
    pass_end();
    cmd_begin();
    vt_layout(vt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    cmd_begin();
    image_barrier(O.col[k], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    {   VkImageBlit b;
        memset(&b, 0, sizeof b);
        b.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; b.srcSubresource.layerCount = 1; b.dstSubresource = b.srcSubresource;
        b.srcOffsets[0].x = s4[0]; b.srcOffsets[0].y = s4[1]; b.srcOffsets[1].x = s4[0] + w; b.srcOffsets[1].y = s4[1] + h; b.srcOffsets[1].z = 1;
        b.dstOffsets[1].x = w; b.dstOffsets[1].y = h; b.dstOffsets[1].z = 1;
        vkCmdBlitImage(R.cmd, vt->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, O.col[k], VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
    }
    uint64_t v = ++O.tl_val;
    O.slot_flow[k] = O.have_prev; O.slot_wait[k] = v;      /* read by the presenter only after the push below (under its lock) */
    if (O.have_prev) {
        cmd_submit(NULL, O.g2of[k]);                /* the frame's own commands and the copy */
        ofx_flow(k);
        cmd_begin();
        image_barrier(O.col[k], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(R.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, O.pipe);
        VkDescriptorImageInfo ii[3] = {{O.smp, O.col_view[p], VK_IMAGE_LAYOUT_GENERAL}, {O.smp, O.col_view[k], VK_IMAGE_LAYOUT_GENERAL}, {NULL, O.mid_view[k], VK_IMAGE_LAYOUT_GENERAL}};
        VkDescriptorBufferInfo bi[2] = {{O.buf[k], 0, VK_WHOLE_SIZE}, {O.stats, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet wr[5];
        memset(wr, 0, sizeof wr);
        for (int i = 0; i < 5; i++) { wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstBinding = (uint32_t)i; wr[i].descriptorCount = 1; }
        wr[0].descriptorType = wr[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wr[0].pImageInfo = &ii[0]; wr[1].pImageInfo = &ii[1];
        wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wr[2].pImageInfo = &ii[2];
        wr[3].descriptorType = wr[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[3].pBufferInfo = &bi[0]; wr[4].pBufferInfo = &bi[1];
        vkCmdPushDescriptorSetKHR(R.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, O.pl, 0, 5, wr);
        int pc[5] = {w, h, O.gw, O.gh, O.stats_on};
        vkCmdPushConstants(R.cmd, O.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        vkCmdDispatch(R.cmd, (uint32_t)(w + 15) / 16, (uint32_t)(h + 7) / 8, 1);
        R.pipe_bound = NULL;
        g_sub_tl = O.tl; g_sub_tl_val = v;
        cmd_submit(O.of2g[k], NULL);
        ofx_push(O.mid[k], v, 0, k);
        ofx_push(O.col[k], v, 1, k);
        InterlockedIncrement64(&O.made);
        {   static long long shot[8];
            static int nshot = -1, ishot;
            if (nshot < 0) { const char *e = getenv("SMO_VK_INTERP_SHOT"); nshot = 0; while (e && *e && nshot < 8) { shot[nshot++] = atoll(e); e = strchr(e, ','); if (e) e++; } }
            if (ishot < nshot && g_play_frames >= shot[ishot]) { ofx_shot(p, k, v); ishot++; }
        }
    } else {
        g_sub_tl = O.tl; g_sub_tl_val = v;
        cmd_submit(NULL, NULL);
        ofx_push(O.col[k], v, 0, k);
    }
    O.have_prev = 1;
    O.k = (k + 1) % OFX_N;
    if (O.stats_on && O.made && O.made % 600 == 0) {
        volatile uint32_t *n = O.stats_mem.map;
        double all = (double)n[0] + n[1] + n[2];
        if (all > 0) printf("nvn_vk_interp: made frames' pixels (last ~600): forward flow %.1f%%, backward flow %.1f%%, no motion %.1f%%\n", 100.0 * n[0] / all, 100.0 * n[1] / all, 100.0 * n[2] / all);
        n[0] = n[1] = n[2] = 0;
    }
    return 1;
}

/* ---- the presenter thread */
static void ofx_wait_until(LONG64 t) {
    for (;;) {
        LONG64 now = tk();
        if (now >= t) return;
        LARGE_INTEGER fq; QueryPerformanceFrequency(&fq);
        double ms = (double)(t - now) * 1000.0 / (double)fq.QuadPart;
        if (ms > 2.0) Sleep((DWORD)(ms - 1.5)); else SwitchToThread();
    }
}
static void ofx_img_barrier(VkCommandBuffer cb, VkImage img, VkFlags ss, VkFlags sa, VkFlags ds, VkFlags da) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; b.subresourceRange.levelCount = 1; b.subresourceRange.layerCount = 1;
    b.srcAccessMask = sa; b.dstAccessMask = da; b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(cb, ss, ds, 0, 0, NULL, 0, NULL, 1, &b);
}
/* presenter: shows src, or (ext_slot >= 0) the frame of that slot moved on by alpha game frames, made here first */
static void ofx_show(VkImage src, uint64_t wait, int ext_slot, float alpha) {
    static int ext_i;
    int i = O.p_i;
    O.p_i = (i + 1) % 4;
    if (O.p_pending[i]) { vkWaitForFences(V.device, 1, &O.p_fence[i], VK_TRUE, 5000000000ull); vkResetFences(V.device, 1, &O.p_fence[i]); O.p_pending[i] = 0; }
    VkSemaphore acq = O.p_acq[O.p_ai];
    O.p_ai = (O.p_ai + 1) % 8;
    uint32_t index = 0;
    VkResult r = vkAcquireNextImageKHR(V.device, V.swapchain, 1000000000ull, acq, NULL, &index);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { InterlockedExchange(&O.swap_bad, 1); InterlockedIncrement64(&O.skipped); return; }
    VkCommandBuffer cb = O.p_cmd[i];
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cb, 0);
    vkBeginCommandBuffer(cb, &bi);
    if (ext_slot >= 0) {
        src = O.ext[ext_i]; ext_i ^= 1;
        ofx_img_barrier(cb, src, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, O.xpipe);
        VkDescriptorImageInfo ii[2] = {{O.smp, O.col_view[ext_slot], VK_IMAGE_LAYOUT_GENERAL}, {NULL, O.ext_view[ext_i ^ 1], VK_IMAGE_LAYOUT_GENERAL}};
        VkDescriptorBufferInfo fb = {O.buf[ext_slot], 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr[3];
        memset(wr, 0, sizeof wr);
        for (int n = 0; n < 3; n++) { wr[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[n].dstBinding = (uint32_t)n; wr[n].descriptorCount = 1; }
        wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wr[0].pImageInfo = &ii[0];
        wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wr[1].pImageInfo = &ii[1];
        wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[2].pBufferInfo = &fb;
        vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, O.xpl, 0, 3, wr);
        struct { int size[2], grid[2]; float alpha; } pc = {{O.w, O.h}, {O.gw, O.gh}, alpha};
        vkCmdPushConstants(cb, O.xpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(cb, (uint32_t)(O.w + 15) / 16, (uint32_t)(O.h + 7) / 8, 1);
        ofx_img_barrier(cb, src, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    }
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = V.images[index];
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; b.subresourceRange.levelCount = 1; b.subresourceRange.layerCount = 1;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    VkImageBlit bl;
    memset(&bl, 0, sizeof bl);
    bl.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; bl.srcSubresource.layerCount = 1; bl.dstSubresource = bl.srcSubresource;
    bl.srcOffsets[1].x = O.w; bl.srcOffsets[1].y = O.h; bl.srcOffsets[1].z = 1;
    bl.dstOffsets[1].x = (int32_t)V.extent.width; bl.dstOffsets[1].y = (int32_t)V.extent.height; bl.dstOffsets[1].z = 1;
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, V.images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    ov_record(cb, V.images[index]);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = 0; b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    vkEndCommandBuffer(cb);
    uint64_t pv = O.pval + 1;
    VkSemaphore waits[2] = {acq, O.tl}, sigs[2] = {O.p_rend[index & 15], O.ptl};
    uint64_t wv[2] = {0, wait}, sv[2] = {0, pv};
    VkFlags stages[2] = {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    VkTimelineSemaphoreSubmitInfo tsi = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    tsi.waitSemaphoreValueCount = 2; tsi.pWaitSemaphoreValues = wv; tsi.signalSemaphoreValueCount = 2; tsi.pSignalSemaphoreValues = sv;
    VkSubmitInfo sb = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    sb.pNext = &tsi; sb.waitSemaphoreCount = 2; sb.pWaitSemaphores = waits; sb.pWaitDstStageMask = stages;
    sb.commandBufferCount = 1; sb.pCommandBuffers = &cb; sb.signalSemaphoreCount = 2; sb.pSignalSemaphores = sigs;
    VkPresentInfoKHR pr = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pr.waitSemaphoreCount = 1; pr.pWaitSemaphores = &sigs[0]; pr.swapchainCount = 1; pr.pSwapchains = &V.swapchain; pr.pImageIndices = &index;
    int shared = O.pqueue == V.queue;                /* only without a second queue: shared with the render thread */
    if (shared) AcquireSRWLockExclusive(&g_qlock);
    VkResult rs = vkQueueSubmit(O.pqueue, 1, &sb, O.p_fence[i]);
    VkResult rp = rs == VK_SUCCESS ? vkQueuePresentKHR(O.pqueue, &pr) : rs;
    if (shared) ReleaseSRWLockExclusive(&g_qlock);
    if (rs == VK_SUCCESS) { O.p_pending[i] = 1; O.pval = pv; }
    if (rp != VK_SUCCESS) InterlockedExchange(&O.swap_bad, 1);
    if (rp == VK_SUCCESS || rp == VK_SUBOPTIMAL_KHR) { InterlockedIncrement64(&O.shown); g_wd_beat = tk(); }
    else { InterlockedIncrement64(&O.skipped); vk_ok("vkQueuePresentKHR (interpolation)", rp); }
}
/* The presenter. A made frame is shown as soon as the GPU has it, the game's frame half a game frame after it. When the next
 * one does not come in time (half a game frame and 2 ms after the last present), the newest frame of the game is shown moved on
 * along its own motion instead, half a game frame further each time, at most ext_max times; the made frame that comes after
 * such predictions lies before them in time and is left out (the game's frame after it is shown). */
static DWORD WINAPI ofx_thread(void *arg) {
    (void)arg;
    LONG64 last = 0;
    int last_slot = -1, n_ext = 0, run = 0;
    os_pin_host_thread();
    LARGE_INTEGER fq;
    QueryPerformanceFrequency(&fq);
    for (;;) {
        OfxItem it;
        int ext = 0;
        EnterCriticalSection(&O.lock);
        for (;;) {
            if (O.qn || O.stop) break;
            double half = O.period * 0.5;
            if (O.ext_on && !O.no_ext && last && last_slot >= 0 && O.slot_flow[last_slot] && n_ext < O.ext_max) {
                LONG64 due = last + (LONG64)((half + 2.0) * (double)fq.QuadPart / 1000.0), now = tk();
                if (now >= due) { ext = 1; break; }
                DWORD ms = (DWORD)((double)(due - now) * 1000.0 / (double)fq.QuadPart);
                SleepConditionVariableCS(&O.cv, &O.lock, ms ? ms : 1);
            } else SleepConditionVariableCS(&O.cv, &O.lock, 100);
        }
        if (!ext && !O.qn) { LeaveCriticalSection(&O.lock); break; }      /* stop */
        if (ext) { it.img = NULL; it.wait = O.slot_wait[last_slot]; it.half = 0; it.slot = last_slot; O.slot_items[last_slot]++; }
        else { it = O.q[O.qh]; O.qh = (O.qh + 1) % 8; O.qn--; }
        O.busy = 1;
        double half = O.period * 0.5;
        LeaveCriticalSection(&O.lock);
        if (ext) {
            n_ext++; run++;
            if (n_ext == 1) InterlockedIncrement64(&O.ext_runs);
            if (!O.swap_bad) { ofx_show(NULL, it.wait, it.slot, 0.5f * (float)n_ext); InterlockedIncrement64(&O.ext_shown); }
            last = tk();
        } else if (!it.half && n_ext) {             /* a made frame from before the predictions: left out */
            InterlockedIncrement64(&O.mid_dropped);
        } else {
            VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wi.semaphoreCount = 1; wi.pSemaphores = &O.tl; wi.pValues = &it.wait;
            vkWaitSemaphores(V.device, &wi, 2000000000ull);
            if (it.half && last) {                  /* the game's frame: half a game frame after the one shown before it */
                LONG64 due = last + (LONG64)(half * (double)fq.QuadPart / 1000.0);
                if (tk() > due + fq.QuadPart / 500) InterlockedIncrement64(&O.late);
                ofx_wait_until(due);
            }
            if (!O.swap_bad) ofx_show(it.img, it.wait, -1, 0.0f); else InterlockedIncrement64(&O.skipped);
            last = tk();
            if (it.half || !O.slot_flow[it.slot]) { last_slot = it.slot; if (run > O.ext_longest) O.ext_longest = run; n_ext = 0; run = 0; }      /* a frame of the game */
        }
        EnterCriticalSection(&O.lock);
        O.p_used[it.slot] = O.pval;
        O.slot_items[it.slot]--;
        O.busy = 0;
        WakeAllConditionVariable(&O.idle_cv);
        LeaveCriticalSection(&O.lock);
    }
    return 0;
}
static void ofx_stop(void) {
    if (!O.thread) return;
    ofx_drain();
    EnterCriticalSection(&O.lock);
    O.stop = 1;
    WakeAllConditionVariable(&O.cv);
    LeaveCriticalSection(&O.lock);
    WaitForSingleObject(O.thread, 5000);
    CloseHandle(O.thread);
    O.thread = NULL;
    printf("nvn_vk_interp: %lld frames made and shown between the game's, %lld presents in all by the presenter, %lld late by more than 2 ms, %lld not shown (swapchain out of date)\n",
           (long long)O.made, (long long)O.shown, (long long)O.late, (long long)O.skipped);
    printf("nvn_vk_interp: late frames of the game bridged %lld times by %lld predicted frames (longest run %lld); made frames left out after a prediction: %lld\n",
           (long long)O.ext_runs, (long long)O.ext_shown, (long long)O.ext_longest, (long long)O.mid_dropped);
    {   LARGE_INTEGER fq; QueryPerformanceFrequency(&fq);
        printf("nvn_vk_interp: the render thread waited for slots %.0f ms in all, longest %.1f ms; the presenter has %s\n", (double)g_ofx_wait_tk * 1000.0 / (double)fq.QuadPart,
               (double)g_ofx_wait_max * 1000.0 / (double)fq.QuadPart, O.pqueue != V.queue ? "a queue of its own" : "the render thread's queue (shared under a lock)");
    }
    fflush(stdout);
}
