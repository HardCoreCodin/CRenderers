
#define HUD_LENGTH 32
#define HUD_LEFT 10
#define HUD_TOP 10
#define HUD_COLOR 0x0000FF00

static char HUD_text[HUD_LENGTH];

char* HUD_template = "___1x___2px\n___3fps\n___4ms\n5__";
char* HUD_char;
char* HUD_width;
char* HUD_height;
char* HUD_fps;
char* HUD_msf;
char* HUD_mode;

void onMouseCaptured() {
    mouse.is_captured = true;
    HUD_char = HUD_mode;
    *HUD_char++ = 'F';
    *HUD_char++ = 'p';
    *HUD_char   = 's';
}

void onMouseUnCaptured() {
    mouse.is_captured = false;
    HUD_char = HUD_mode;
    *HUD_char++ = 'O';
    *HUD_char++ = 'r';
    *HUD_char   = 'b';
}

void init_hud() {
    frame_buffer.pixels = (u32*)allocate_memory(RENDER_SIZE);

    char* HUD_text_char = HUD_text;
    HUD_char = HUD_template;
    while (*HUD_char) {
        switch (*HUD_char) {
            case '1':  HUD_width = HUD_text_char; break;
            case '2':  HUD_height = HUD_text_char; break;
            case '3':  HUD_fps = HUD_text_char; break;
            case '4':  HUD_msf = HUD_text_char; break;
            case '5':  HUD_mode = HUD_text_char; break;
        }

        *HUD_text_char++ = *HUD_char++;
    }
    *HUD_text_char = '\0';

    onMouseUnCaptured();
}