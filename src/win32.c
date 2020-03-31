#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "lib/core/string.h"
#include "lib/render/hud.h"
#include "lib/render/draw.h"
#include "lib/render/engines/ray_tracer/engine.h"

#define RAW_INPUT_MAX_SIZE Kilobytes(1)
#define MEMORY_SIZE Gigabytes(1)
#define MEMORY_BASE Terabytes(2)

static bool is_running;
static f32 dx, dy;
static f64 ticks_per_second, seconds_per_tick, milliseconds_per_tick, delta_time;

static WNDCLASSA window_class;
static HWND window;
static HDC win_dc;
static BITMAPINFO info;
static RECT win_rect;
static LARGE_INTEGER current_frame_ticks, last_frame_ticks, delta_ticks;
static RAWINPUTDEVICE rid;
static RAWINPUT* raw_inputs;
static RAWMOUSE raw_mouse;
static UINT size_ri, size_rih = sizeof(RAWINPUTHEADER);

inline void resizeFrameBuffer() {
    GetClientRect(window, &win_rect);

    info.bmiHeader.biWidth = win_rect.right - win_rect.left;
    info.bmiHeader.biHeight = win_rect.top - win_rect.bottom;

    frame_buffer.width = (u16)info.bmiHeader.biWidth;
    frame_buffer.height = (u16)-info.bmiHeader.biHeight;
    frame_buffer.size = frame_buffer.width * frame_buffer.height;

    printNumberIntoString(frame_buffer.width, hud.width);
    printNumberIntoString(frame_buffer.height, hud.height);
    onFrameBufferResized(frame_buffer.width, frame_buffer.height);
}

void updateAndRender() {
    last_frame_ticks = current_frame_ticks;
    QueryPerformanceCounter(&current_frame_ticks);
    delta_ticks.QuadPart = current_frame_ticks.QuadPart - last_frame_ticks.QuadPart;
    delta_time = delta_ticks.QuadPart * seconds_per_tick;
    update((f32)delta_time);
    render(&frame_buffer);
    InvalidateRgn(window, NULL, FALSE);
    UpdateWindow(window);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_DESTROY:
            is_running = false;
            PostQuitMessage(0);
            break;

        case WM_SIZE:
            resizeFrameBuffer();
            updateAndRender();
            break;

        case WM_PAINT:
            if (hud.is_visible)
                drawText(hud.text, HUD_COLOR, HUD_LEFT, HUD_TOP);

            SetDIBitsToDevice(win_dc,
                    0, 0, frame_buffer.width, frame_buffer.height,
                    0, 0, 0, frame_buffer.height,
                    frame_buffer.pixels, &info, DIB_RGB_COLORS);

            ValidateRgn(window, NULL);

            break;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            switch ((u32)wParam) {
                case 'W': controls.keyboard.pressed |= controls.buttons.FORWARD; break;
                case 'A': controls.keyboard.pressed |= controls.buttons.LEFT; break;
                case 'S': controls.keyboard.pressed |= controls.buttons.BACKWARD; break;
                case 'D': controls.keyboard.pressed |= controls.buttons.RIGHT; break;
                case 'R': controls.keyboard.pressed |= controls.buttons.UP; break;
                case 'F': controls.keyboard.pressed |= controls.buttons.DOWN; break;

                case VK_SPACE:
                    if (controls.mouse.is_captured) {
                        controls.mouse.is_captured = false;
                        setControllerModeInHUD(false, hud.mode);
                        ReleaseCapture();
                        ShowCursor(true);
                    } else {
                        controls.mouse.is_captured = true;
                        setControllerModeInHUD(true, hud.mode);
                        SetCapture(window);
                        ShowCursor(false);
                    }
                    break;

                case VK_TAB:
                    hud.is_visible =! hud.is_visible;
                    break;

                case VK_ESCAPE:
                    is_running = false;
                    break;
            }
            break;

        case WM_SYSKEYUP:
        case WM_KEYUP:
            switch ((u32)wParam) {
                case 'W': controls.keyboard.pressed &= (u8)~controls.buttons.FORWARD; break;
                case 'A': controls.keyboard.pressed &= (u8)~controls.buttons.LEFT; break;
                case 'S': controls.keyboard.pressed &= (u8)~controls.buttons.BACKWARD; break;
                case 'D': controls.keyboard.pressed &= (u8)~controls.buttons.RIGHT; break;
                case 'R': controls.keyboard.pressed &= (u8)~controls.buttons.UP; break;
                case 'F': controls.keyboard.pressed &= (u8)~controls.buttons.DOWN; break;
            }
            break;

        case WM_MOUSEWHEEL:
            onMouseWheelChanged(GET_WHEEL_DELTA_WPARAM(wParam) / 120.0f);
            break;

        case WM_INPUT:
            size_ri = 0;
            if (!GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &size_ri, size_rih) && size_ri &&
                 GetRawInputData((HRAWINPUT)lParam, RID_INPUT, raw_inputs, &size_ri, size_rih) == size_ri &&
                 raw_inputs->header.dwType == RIM_TYPEMOUSE) {
                raw_mouse = raw_inputs->data.mouse;

                if (     raw_mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN) controls.mouse.pressed |= controls.buttons.LEFT;
                else if (raw_mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP)   controls.mouse.pressed &= (u8)~controls.buttons.LEFT;;

                if (     raw_mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN) controls.mouse.pressed |= controls.buttons.RIGHT;
                else if (raw_mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP)   controls.mouse.pressed &= (u8)~controls.buttons.RIGHT;;

                if (     raw_mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN) controls.mouse.pressed |= controls.buttons.MIDDLE;
                else if (raw_mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP)   controls.mouse.pressed &= (u8)~controls.buttons.MIDDLE;;

                dx = (f32)raw_mouse.lLastX;
                dy = (f32)raw_mouse.lLastY;

                if (dx || dy)
                    onMousePositionChanged(dx, dy);
            }

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }

    return 0;
}

int APIENTRY WinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPSTR     lpCmdLine,
                     int       nCmdShow) {
    LARGE_INTEGER performance_frequency;
    QueryPerformanceFrequency(&performance_frequency);
    ticks_per_second = (f64)performance_frequency.QuadPart;
    seconds_per_tick = 1.0f / ticks_per_second;
    milliseconds_per_tick = 1000.0f / ticks_per_second;

    f64 ticks_per_interval = ticks_per_second / 4;;
    // Initialize the memory:
    memory.address = (u8*)VirtualAlloc(
            (LPVOID)MEMORY_BASE,
            MEMORY_SIZE,
            MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    if (!memory.address)
        return -1;

    initHUD();
    initControls(&controls);
    initFrameBuffer();
    initRenderEngine();

    info.bmiHeader.biSize        = sizeof(info.bmiHeader);
    info.bmiHeader.biCompression = BI_RGB;
    info.bmiHeader.biBitCount    = 32;
    info.bmiHeader.biPlanes      = 1;

    window_class.lpszClassName  = "RnDer";
    window_class.hInstance      = hInstance;
    window_class.lpfnWndProc    = WndProc;
    window_class.style          = CS_OWNDC|CS_HREDRAW|CS_VREDRAW;
    window_class.hCursor        = LoadCursorA(0, IDC_ARROW);

    RegisterClassA(&window_class);

    window = CreateWindowA(
            window_class.lpszClassName,
            TITLE,
            WS_OVERLAPPEDWINDOW,

            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,

            0,
            0,
            hInstance,
            0
    );
    if (!window)
        return -1;

    raw_inputs = (RAWINPUT*)allocate(RAW_INPUT_MAX_SIZE);

    rid.usUsagePage = 0x01;
    rid.usUsage = 0x02;
    if (!RegisterRawInputDevices(&rid, 1, sizeof(rid)))
        return -1;

    win_dc = GetDC(window);
    QueryPerformanceCounter(&current_frame_ticks);
    ShowWindow(window, nCmdShow);

    MSG message;
    u8 frames = 0;
    f64 ticks = 0;

    LARGE_INTEGER performance_counter;
    f64 prior_frame_counter = 0;
    f64 current_frame_counter = 0;

    is_running = true;

    while (is_running) {
        while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageA(&message);
        }

        updateAndRender();

        if (hud.is_visible) {
            if (prior_frame_counter == 0) {
                QueryPerformanceCounter(&performance_counter);
                prior_frame_counter = (f64)performance_counter.QuadPart;
                continue;
            }

            prior_frame_counter = current_frame_counter;
            QueryPerformanceCounter(&performance_counter);
            current_frame_counter = (f64)performance_counter.QuadPart;
            ticks += current_frame_counter - prior_frame_counter;
            frames++;

            if (ticks >= ticks_per_interval) {
                printNumberIntoString((u16)(frames / ticks * ticks_per_second), hud.fps);
                printNumberIntoString((u16)(ticks / frames * milliseconds_per_tick), hud.msf);

                ticks = frames = 0;
            }
        } else
            prior_frame_counter = 0;
    }

    return (int)message.wParam;
}