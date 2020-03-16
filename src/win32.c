#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "ray_trace.h"

static char title_string[32];
static f64 counts_per_second;

static HWND window;
static HDC device_context;
static RECT rect = {0, 0, INITIAL_WIDTH, INITIAL_HEIGHT};
static BITMAPINFO info;
static POINT current_mouse_position, prior_mouse_position;

inline void resizeFrameBuffer() {
    GetClientRect(window, &rect);
    info.bmiHeader.biWidth = frame_buffer.width = (u16)(rect.right - rect.left);
    info.bmiHeader.biHeight = frame_buffer.height = (u16)(rect.bottom - rect.top);
    frame_buffer.size = frame_buffer.width * frame_buffer.height;
    onFrameBufferResized();
}

void updateAndRender() {
    static LARGE_INTEGER current_frame_time;
    LARGE_INTEGER last_frame_time = current_frame_time;
    QueryPerformanceCounter(&current_frame_time);
    update((f32)(
        (f64)(
            current_frame_time.QuadPart - last_frame_time.QuadPart
        ) / counts_per_second
    ));

    LARGE_INTEGER before_rendering, after_rendering;
    QueryPerformanceCounter(&before_rendering);
    render();
    QueryPerformanceCounter(&after_rendering);

//    printTitleIntoString(
//            frame_buffer.width,
//            frame_buffer.height,
//            (u16)(
//                counts_per_second / (f64)(
//                    after_rendering.QuadPart - before_rendering.QuadPart
//                )),
//            "RT",
//            title_string);
//    SetWindowTextA(window, title_string);

//    RedrawWindow(window, NULL, NULL, RDW_INVALIDATE|RDW_NOCHILDREN|RDW_UPDATENOW);
    InvalidateRgn(window, NULL, false);
    UpdateWindow(window);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_DESTROY:
            app.is_running = false;
            PostQuitMessage(0);
            break;
//
//        case WM_ERASEBKGND:
//            return 1;

        case WM_SIZE:
            resizeFrameBuffer();
            updateAndRender();
            break;

        case WM_PAINT:
            SetDIBitsToDevice(
                    device_context, 0, 0,
                    frame_buffer.width,
                    frame_buffer.height, 0, 0, 0,
                    frame_buffer.height,
                    frame_buffer.pixels,
                    &info,
                    DIB_RGB_COLORS);
            ValidateRgn(window, NULL);
//            RedrawWindow(window, NULL, NULL, RDW_VALIDATE|RDW_NOERASE);
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
                mouse.is_captured = false;
                ReleaseCapture();
                ShowCursor(true);
            } else {
                mouse.is_captured = true;
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

int APIENTRY WinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    LARGE_INTEGER performance_frequency;
    QueryPerformanceFrequency(&performance_frequency);
    counts_per_second = (f64)performance_frequency.QuadPart;

    // Initialize the memory:
    memory.address = (u8*)VirtualAlloc((LPVOID)memory.base, memory.size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    if (!memory.address)
        return -1;

    init_core();
    init_renderer();

    info.bmiHeader.biSize = sizeof(info.bmiHeader);
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;

    WNDCLASSA window_class;
    window_class.lpszClassName  = "RnDer";
    window_class.hInstance      = hInstance;
    window_class.lpfnWndProc    = WndProc;
    window_class.style          = CS_OWNDC|CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS;
//    window_class.hbrBackground  = (HBRUSH)COLOR_WINDOW+1;
    window_class.hCursor        = LoadCursorA(NULL, IDC_ARROW);
    window_class.hbrBackground  = NULL;
//    window_class.hCursor        = NULL;
    window_class.hIcon          = NULL;
    window_class.lpszMenuName   = NULL;
    window_class.cbClsExtra     = 0;
    window_class.cbWndExtra     = 0;

    RegisterClassA(&window_class);

    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, false);
    window = CreateWindowA(
            window_class.lpszClassName,
            TITLE,
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            NULL,
            NULL,
            hInstance,
            NULL
    );
    if (!window)
        return -1;

    device_context = GetDC(window);  //GetDCEx(window, NULL, DCX_WINDOW);

    ShowWindow(window, nCmdShow);
    GetCursorPos(&current_mouse_position);
    MSG message;

    while (app.is_running) {
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
    }

    return (int)message.wParam;
}