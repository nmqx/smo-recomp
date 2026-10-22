/* nvn_vk_overlay.c - the frame-rate counter drawn into the picture, top right (graphics.ini frame_time_overlay = on).
 * Included by nvn_vk_draw.c. Two numbers: the frames the game made in the last half second and the pictures shown (with
 * frame interpolation, about twice as many), as "GAME 58  SHOWN 116". The text is drawn on the CPU into a small
 * host-visible buffer (a 5x7 font, scaled 3x) and copied into the swapchain image after its picture, in the present's own
 * command buffer: no shader, no pass. A ring of buffers so that a copy still in flight is never written. */

#define OV_SCALE 3
#define OV_W 300
#define OV_H 31
#define OV_RING 8
static struct { int on, made, i; VkBuffer buf[OV_RING]; VMem mem[OV_RING]; volatile LONG64 game, shown; LONG64 t0, g0, s0; int fps_game, fps_shown; } OV;

static const uint8_t k_ov_font[][7] = {        /* 5x7, bit 4 = leftmost column */
    {14, 17, 19, 21, 25, 17, 14}, {4, 12, 4, 4, 4, 4, 14}, {14, 17, 1, 2, 4, 8, 31}, {31, 2, 4, 2, 1, 17, 14},
    {2, 6, 10, 18, 31, 2, 2}, {31, 16, 30, 1, 1, 17, 14}, {6, 8, 16, 30, 17, 17, 14}, {31, 1, 2, 4, 8, 8, 8},
    {14, 17, 17, 14, 17, 17, 14}, {14, 17, 17, 15, 1, 2, 12},              /* 0..9 */
    {14, 17, 16, 23, 17, 17, 15}, {14, 17, 17, 31, 17, 17, 17}, {17, 27, 21, 21, 17, 17, 17}, {31, 16, 16, 30, 16, 16, 31},  /* G A M E */
    {15, 16, 16, 14, 1, 1, 30}, {17, 17, 17, 31, 17, 17, 17}, {14, 17, 17, 17, 17, 17, 14}, {17, 17, 17, 21, 21, 21, 10},    /* S H O W */
    {17, 25, 21, 19, 17, 17, 17}, {0, 0, 0, 0, 0, 0, 0}                   /* N, space */
};
static int ov_glyph(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    switch (ch) { case 'G': return 10; case 'A': return 11; case 'M': return 12; case 'E': return 13; case 'S': return 14; case 'H': return 15; case 'O': return 16; case 'W': return 17; case 'N': return 18; }
    return 19;
}
static int ov_enabled(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("SMO_FPS_OVERLAY"); on = e && e[0] ? atoi(e) != 0 : gfx_options()->overlay; }
    return on;
}
static void ov_count_game(void) { if (ov_enabled()) InterlockedIncrement64(&OV.game); }      /* vk_present2: a frame of the game */
/* Copies the counter into `img` (the swapchain image index's, in TRANSFER_DST layout, its picture already blitted). Any
 * thread that records a present; the ring index is atomic. */
static void ov_record(VkCommandBuffer cb, VkImage img) {
    if (!ov_enabled() || V.extent.width < OV_W + 32 || V.extent.height < OV_H + 32) return;
    InterlockedIncrement64(&OV.shown);
    LARGE_INTEGER fq, now;
    QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&now);
    if (!OV.t0) { OV.t0 = now.QuadPart; OV.g0 = OV.game; OV.s0 = OV.shown; }
    double dt = (double)(now.QuadPart - OV.t0) / (double)fq.QuadPart;
    if (dt >= 0.5) {
        OV.fps_game = (int)((double)(OV.game - OV.g0) / dt + 0.5); OV.fps_shown = (int)((double)(OV.shown - OV.s0) / dt + 0.5);
        OV.t0 = now.QuadPart; OV.g0 = OV.game; OV.s0 = OV.shown;
    }
    if (!OV.made) {
        for (int i = 0; i < OV_RING; i++) OV.buf[i] = vbuf_new((uint64_t)OV_W * OV_H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &OV.mem[i]);
        OV.made = 1;
    }
    int bi = (int)(InterlockedIncrement((volatile LONG *)&OV.i) % OV_RING);
    uint32_t *px = (uint32_t *)OV.mem[bi].map;
    char text[40];
    snprintf(text, sizeof text, "GAME %d  SHOWN %d", OV.fps_game, OV.fps_shown);
    int game_ok = OV.fps_game >= 58, game_mid = OV.fps_game >= 45;
    uint32_t bg = 0xff101010u, fg = game_ok ? 0xff40ff40u : game_mid ? 0xffffe040u : 0xffff4040u;      /* 0xAARRGGBB (B8G8R8A8 in memory): green, yellow, red */
    if (V.format.format == VK_FORMAT_R8G8B8A8_UNORM || V.format.format == VK_FORMAT_R8G8B8A8_SRGB) fg = (fg & 0xff00ff00u) | ((fg & 0xff) << 16) | ((fg >> 16) & 0xff);
    for (int k = 0; k < OV_W * OV_H; k++) px[k] = bg;
    int len = (int)strlen(text), x0 = OV_W - 5 - len * 6 * OV_SCALE;
    for (int c = 0; c < len; c++) {
        const uint8_t *g = k_ov_font[ov_glyph(text[c])];
        for (int r = 0; r < 7; r++) for (int col = 0; col < 5; col++) if (g[r] >> (4 - col) & 1)
            for (int dy = 0; dy < OV_SCALE; dy++) for (int dx = 0; dx < OV_SCALE; dx++) {
                int x = x0 + (c * 6 + col) * OV_SCALE + dx, y = 5 + r * OV_SCALE + dy;
                if (x >= 0 && x < OV_W && y < OV_H) px[y * OV_W + x] = fg;
            }
    }
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};      /* after the blit that wrote the picture (the same pixels) */
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkBufferImageCopy rg;
    memset(&rg, 0, sizeof rg);
    rg.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; rg.imageSubresource.layerCount = 1;
    rg.imageOffset.x = (int32_t)V.extent.width - OV_W - 16; rg.imageOffset.y = 16;
    rg.imageExtent.width = OV_W; rg.imageExtent.height = OV_H; rg.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(cb, OV.buf[bi], img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rg);
}
