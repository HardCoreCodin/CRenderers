#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <rpc.h>
#include "ray_trace.h"

#define RAW_INPUT_MAX_SIZE Kilobytes(1)

static f64 ticks_per_second, seconds_per_tick, milliseconds_per_tick;
static f32 dx, dy;

static WNDCLASSA window_class;
static HWND window;
static HDC win_dc, ovr_dc;
static HBITMAP ovr_bm, dib_bm;
static BITMAPINFO info;
static RECT win_rect, ovr_rect;
static LARGE_INTEGER current_frame_ticks, last_frame_ticks;
static PAINTSTRUCT ps;
static RAWINPUTDEVICE rid;
static RAWINPUT* raw_inputs;
static RAWMOUSE raw_mouse;
static UINT size_ri, size_rih = sizeof(RAWINPUTHEADER);
static HFONT font;
//
//inline void resizeFrameBuffer() {
//    GetClientRect(window, &win_rect);
//
//    info.bmiHeader.biWidth = win_rect.right - win_rect.left;
//    info.bmiHeader.biHeight = win_rect.bottom - win_rect.top;
//
//    HDC hdc = GetDC(window);
//    if (dib_bm) DeleteObject(dib_bm);
//    dib_bm = CreateDIBSection(
//            hdc,
//            &info,
//            DIB_RGB_COLORS,
//            (void**)&frame_buffer.pixels,
//            0,
//            0);
//    ReleaseDC(window, hdc);
//
//    frame_buffer.width = (u16)info.bmiHeader.biWidth;
//    frame_buffer.height = (u16)info.bmiHeader.biHeight;
//    frame_buffer.size = frame_buffer.width * frame_buffer.height;
//
//    printNumberIntoString(frame_buffer.width, RESOLUTION.string + RESOLUTION.n1);
//    printNumberIntoString(frame_buffer.height, RESOLUTION.string + RESOLUTION.n2);
//
//    onFrameBufferResized();
//}

void refreshDisplay() {
    HDC hdc = GetDC(window);
    HDC mdc = CreateCompatibleDC(hdc);

    SetBkMode(mdc, TRANSPARENT);
    SelectObject(mdc, dib_bm);
    BitBlt(hdc, 0, 0, frame_buffer.width, frame_buffer.height,
           mdc, 0, 0, SRCCOPY);

    SelectObject(mdc, font);
    SetTextColor(mdc, 0x0000FF00);
        TextOutA(mdc, OVR_LEFT, RESOLUTION.y, RESOLUTION.string, RESOLUTION.length);
        TextOutA(mdc, OVR_LEFT, FRAME_RATE.y, FRAME_RATE.string, FRAME_RATE.length);
        TextOutA(mdc, OVR_LEFT, FRAME_TIME.y, FRAME_TIME.string, FRAME_TIME.length);
        TextOutA(mdc, OVR_LEFT, NAVIGATION.y, NAVIGATION.string, NAVIGATION.length);

    BitBlt(hdc, 0, 0, frame_buffer.width, frame_buffer.height,
           mdc, 0, 0, SRCPAINT);

    DeleteDC(mdc);
    ReleaseDC(window, hdc);
}

void updateAndRender() {
    last_frame_ticks = current_frame_ticks;
    QueryPerformanceCounter(&current_frame_ticks);
    update((f32)(seconds_per_tick * (f64)(current_frame_ticks.QuadPart - last_frame_ticks.QuadPart)));
    render();
    refreshDisplay();
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_DESTROY:
            app.is_running = false;
            PostQuitMessage(0);
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

                case VK_SPACE:
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

                case VK_TAB:
                    app.is_HUD_visible =! app.is_HUD_visible;
                    break;

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

        case WM_MOUSEWHEEL:
            onMouseWheelChanged(GET_WHEEL_DELTA_WPARAM(wParam) / 120.0f);
            break;

        case WM_INPUT:
            size_ri = 0;
            if (!GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &size_ri, size_rih) && size_ri &&
                 GetRawInputData((HRAWINPUT)lParam, RID_INPUT, raw_inputs, &size_ri, size_rih) == size_ri &&
                 raw_inputs->header.dwType == RIM_TYPEMOUSE) {
                raw_mouse = raw_inputs->data.mouse;

                if (     raw_mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN) mouse.pressed |= LEFT;
                else if (raw_mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP)   mouse.pressed &= (u8)~LEFT;;

                if (     raw_mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN) mouse.pressed |= RIGHT;
                else if (raw_mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP)   mouse.pressed &= (u8)~RIGHT;;

                if (     raw_mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN) mouse.pressed |= MIDDLE;
                else if (raw_mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP)   mouse.pressed &= (u8)~MIDDLE;;

                dx = (f32)raw_mouse.lLastX;
                dy = (f32)raw_mouse.lLastY;

                if ((dx || dy) && (mouse.is_captured || (mouse.pressed & RIGHT || mouse.pressed & MIDDLE)))
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
    window_class.style          = CS_BYTEALIGNCLIENT|CS_HREDRAW|CS_VREDRAW;
    window_class.hCursor        = LoadCursorA(0, IDC_ARROW);

    RegisterClassA(&window_class);

    window = CreateWindowA(
            window_class.lpszClassName,
            TITLE,
            WS_OVERLAPPEDWINDOW,

            CW_USEDEFAULT,
            CW_USEDEFAULT,
            300,
            200,

            0,
            0,
            hInstance,
            0
    );
    if (!window)
        return -1;

    raw_inputs = (RAWINPUT*)allocate_memory(RAW_INPUT_MAX_SIZE);

    rid.usUsagePage = 0x01;
    rid.usUsage = 0x02;
    if (!RegisterRawInputDevices(&rid, 1, sizeof(rid)))
        return -1;

    GetClientRect(window, &win_rect);

    info.bmiHeader.biWidth = win_rect.right - win_rect.left;
    info.bmiHeader.biHeight = win_rect.bottom - win_rect.top;

    HDC hdc = GetDC(window);
    dib_bm = CreateDIBSection(
            hdc,
            &info,
            DIB_RGB_COLORS,
            (void**)&frame_buffer.pixels,
            0,
            0);
    ReleaseDC(window, hdc);

    frame_buffer.width = (u16)info.bmiHeader.biWidth;
    frame_buffer.height = (u16)info.bmiHeader.biHeight;
    frame_buffer.size = frame_buffer.width * frame_buffer.height;

    printNumberIntoString(frame_buffer.width, RESOLUTION.string + RESOLUTION.n1);
    printNumberIntoString(frame_buffer.height, RESOLUTION.string + RESOLUTION.n2);

    onFrameBufferResized();

    font = GetStockObject(SYSTEM_FONT);
    ShowWindow(window, nCmdShow);

    MSG message;
    u8 frames = 0;
    f64 ticks = 0;

    LARGE_INTEGER performance_counter;
    f64 prior_frame_counter = 0;
    f64 current_frame_counter = 0;

    while (app.is_running) {
        while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageA(&message);
        }

        updateAndRender();

        if (app.is_HUD_visible) {
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
                printNumberIntoString((u16)(frames / ticks * ticks_per_second), FRAME_RATE.string + FRAME_RATE.n1);
                printNumberIntoString((u16)(ticks / frames * milliseconds_per_tick), FRAME_TIME.string + FRAME_TIME.n1);

                ticks = frames = 0;
            }
        } else
            prior_frame_counter = 0;
    }

    return (int)message.wParam;
}