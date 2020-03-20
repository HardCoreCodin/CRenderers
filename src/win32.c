#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "ray_trace.h"

static f64 ticks_per_second, seconds_per_tick, milliseconds_per_tick;

static WNDCLASSA window_class;
static HWND window;
static HDC win_dc, ovr_dc;
static HBITMAP ovr_bp;
static BITMAPINFO info;
static RECT win_rect, ovr_rect;
static POINT current_mouse_position, prior_mouse_position;
static LARGE_INTEGER current_frame_ticks, last_frame_ticks;
static PAINTSTRUCT ps;

inline void resizeFrameBuffer() {
    GetClientRect(window, &win_rect);

    info.bmiHeader.biWidth = win_rect.right - win_rect.left;
    info.bmiHeader.biHeight = win_rect.bottom - win_rect.top;

    frame_buffer.width = (u16)info.bmiHeader.biWidth;
    frame_buffer.height = (u16)info.bmiHeader.biHeight;
    frame_buffer.size = frame_buffer.width * frame_buffer.height;

    printNumberIntoString(frame_buffer.width, RESOLUTION.string + RESOLUTION.n1);
    printNumberIntoString(frame_buffer.height, RESOLUTION.string + RESOLUTION.n2);

    onFrameBufferResized();
}

void updateAndRender() {
    last_frame_ticks = current_frame_ticks;
    QueryPerformanceCounter(&current_frame_ticks);
    update((f32)(seconds_per_tick * (f64)(current_frame_ticks.QuadPart - last_frame_ticks.QuadPart)));
    render();
    InvalidateRgn(window, 0, false);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_DESTROY:
            app.is_running = false;
            PostQuitMessage(0);
            break;

        case WM_SIZE:
            resizeFrameBuffer();
            updateAndRender();
            break;

        case WM_PAINT:
            BeginPaint(window, &ps);
            SetDIBitsToDevice(
                    win_dc, 0, 0,
                    frame_buffer.width,
                    frame_buffer.height, 0, 0, 0,
                    frame_buffer.height,
                    frame_buffer.pixels,
                    &info,
                    DIB_RGB_COLORS);

            FillRect(ovr_dc, &ovr_rect, (HBRUSH) (COLOR_BACKGROUND + 1));
            TextOutA(ovr_dc, OVR_LEFT, RESOLUTION.y, RESOLUTION.string, RESOLUTION.length);
            TextOutA(ovr_dc, OVR_LEFT, FRAME_RATE.y, FRAME_RATE.string, FRAME_RATE.length);
            TextOutA(ovr_dc, OVR_LEFT, FRAME_TIME.y, FRAME_TIME.string, FRAME_TIME.length);
            TextOutA(ovr_dc, OVR_LEFT, NAVIGATION.y, NAVIGATION.string, NAVIGATION.length);
            BitBlt(win_dc, OVR_LEFT, OVR_TOP, OVR_WIDTH, OVR_HEIGHT, ovr_dc, OVR_LEFT, OVR_TOP, SRCPAINT);

            EndPaint(window, &ps);
            break;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            switch ((u32)wParam) {
                case 'W': keyboard.pressed |= FORWARD; break;
                case 'A': keyboard.pressed |= LEFT; break;
                case 'S': keyboard.pressed |= BACKWARD; break;
                case 'D': keyboard.pressed |= RIGHT; break;
                case 'R': keyboard.pressed |= UP; break;
                case 'F': keyboard.pressed |= DOWN; break;

                case VK_ESCAPE:
                    app.is_running = false;
                    break;
            }
            break;

        case WM_SYSKEYUP:
        case WM_KEYUP:
            switch ((u32)wParam) {
                case 'W': keyboard.pressed &= (u8)~FORWARD; break;
                case 'A': keyboard.pressed &= (u8)~LEFT; break;
                case 'S': keyboard.pressed &= (u8)~BACKWARD; break;
                case 'D': keyboard.pressed &= (u8)~RIGHT; break;
                case 'R': keyboard.pressed &= (u8)~UP; break;
                case 'F': keyboard.pressed &= (u8)~DOWN; break;
            }
            break;

        case WM_LBUTTONDOWN: mouse.pressed |= LEFT; break;
        case WM_RBUTTONDOWN: mouse.pressed |= RIGHT; break;
        case WM_MBUTTONDOWN: mouse.pressed |= MIDDLE; break;

        case WM_LBUTTONUP: mouse.pressed &= (u8)~LEFT; break;
        case WM_RBUTTONUP: mouse.pressed &= (u8)~RIGHT; break;
        case WM_MBUTTONUP: mouse.pressed &= (u8)~MIDDLE; break;

        case WM_LBUTTONDBLCLK:
            if (mouse.is_captured) {
                onMouseUnCaptured();
                ReleaseCapture();
                ShowCursor(true);
            } else {
                onMouseCaptured();
                SetCapture(window);
                ShowCursor(false);
            }
            break;

        case WM_MOUSEWHEEL:
            onMouseWheelChanged(GET_WHEEL_DELTA_WPARAM(wParam) / 120.0f);
            break;

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
    memory.address = (u8*)VirtualAlloc((LPVOID)memory.base, memory.size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    if (!memory.address)
        return -1;

    init_core();
    init_renderer();

    ovr_rect.left = OVR_LEFT;
    ovr_rect.right = OVR_LEFT + OVR_WIDTH;
    ovr_rect.top = OVR_TOP;
    ovr_rect.bottom = OVR_TOP + OVR_HEIGHT;

    info.bmiHeader.biSize        = sizeof(info.bmiHeader);
    info.bmiHeader.biCompression = BI_RGB;
    info.bmiHeader.biBitCount    = 32;
    info.bmiHeader.biPlanes      = 1;

    window_class.lpszClassName  = "RnDer";
    window_class.hInstance      = hInstance;
    window_class.lpfnWndProc    = WndProc;
    window_class.style          = CS_OWNDC|CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS;
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

    win_dc = GetDC(window);  //GetDCEx(window, NULL, DCX_WINDOW);
    GetClientRect(window, &win_rect);
    SetBkMode(win_dc, TRANSPARENT);

    ovr_dc = CreateCompatibleDC(win_dc);
    ovr_bp = CreateCompatibleBitmap(win_dc, OVR_WIDTH, OVR_HEIGHT);
    HFONT ovr_font = GetStockObject(SYSTEM_FONT);
    SelectObject(ovr_dc, ovr_bp);
    SelectObject(ovr_dc, ovr_font);
    SetTextColor(ovr_dc, 0x0000FF00);
    SetBkMode(ovr_dc, TRANSPARENT);

    ShowWindow(window, nCmdShow);

    GetCursorPos(&current_mouse_position);
    MSG message;
    u8 frames = 0;

    LARGE_INTEGER start_frame_counter, end_frame_counter;
    f64 ticks = 0;

    while (app.is_running) {
        QueryPerformanceCounter(&start_frame_counter);

        while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageA(&message);
        }

        prior_mouse_position = current_mouse_position;
        GetCursorPos(&current_mouse_position);
        f32 dx = (f32)(current_mouse_position.x - prior_mouse_position.x);
        f32 dy = (f32)(current_mouse_position.y - prior_mouse_position.y);
        if (dx || dy)
            onMousePositionChanged(dx, dy);

        updateAndRender();

        QueryPerformanceCounter(&end_frame_counter);
        ticks += (f64)(end_frame_counter.QuadPart - start_frame_counter.QuadPart);
        frames++;

        if (ticks >= ticks_per_interval) {
            printNumberIntoString((u16)(frames / ticks * ticks_per_second), FRAME_RATE.string + FRAME_RATE.n1);
            printNumberIntoString((u16)(ticks / frames * milliseconds_per_tick), FRAME_TIME.string + FRAME_TIME.n1);

            ticks = frames = 0;
        }
    }

    return (int)message.wParam;
}