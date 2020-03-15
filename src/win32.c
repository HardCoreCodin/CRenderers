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

    InvalidateRgn(window, NULL, FALSE);
    UpdateWindow(window);
}

void processMessages() {
    MSG message;

    while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE)) {
        switch(message.message) {
            case WM_SYSKEYDOWN:
            case WM_KEYDOWN:
                switch ((u32)message.wParam) {
                    case 'W': keyboard.pressed |= FORWARD; break;
                    case 'A': keyboard.pressed |= LEFT; break;
                    case 'S': keyboard.pressed |= BACKWARD; break;
                    case 'D': keyboard.pressed |= RIGHT; break;
                    case 'R': keyboard.pressed |= UP; break;
                    case 'F': keyboard.pressed |= DOWN; break;

                    case VK_ESCAPE:
                        app.is_running = FALSE;
                        break;
                }
                break;

            case WM_SYSKEYUP:
            case WM_KEYUP:
                switch ((u32)message.wParam) {
                    case 'W': keyboard.pressed &= (u8)~FORWARD; break;
                    case 'A': keyboard.pressed &= (u8)~LEFT; break;
                    case 'S': keyboard.pressed &= (u8)~BACKWARD; break;
                    case 'D': keyboard.pressed &= (u8)~RIGHT; break;
                    case 'R': keyboard.pressed &= (u8)~UP; break;
                    case 'F': keyboard.pressed &= (u8)~DOWN; break;
                }
                break;

            case WM_LBUTTONDBLCLK:
                if (app.is_active) {
                    app.is_active = FALSE;
                    ReleaseCapture();
                } else {
                    app.is_active = TRUE;
                    GetCursorPos(&current_mouse_position);
                    SetCapture(window);
                }
                break;

            case WM_MOUSEWHEEL:
                onMouseWheelChanged(GET_WHEEL_DELTA_WPARAM(message.wParam) / 120.0f);
                break;

            default:
                TranslateMessage(&message);
                DispatchMessageA(&message);
        }
    }


}
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
//        case WM_SYSCOMMAND:
//            switch (wParam)
//            {
//                case SC_MINIMIZE:
//                {
//                    //Do whatever you want!
//                    return TRUE;
//                }
//            }
//        break;

//        case WM_ERASEBKGND:
//            return 1;

        case WM_MOUSEMOVE:
            if (app.is_active) {
                prior_mouse_position = current_mouse_position;
                GetCursorPos(&current_mouse_position);
                f32 dx = (f32)(current_mouse_position.x - prior_mouse_position.x);
                f32 dy = (f32)(current_mouse_position.y - prior_mouse_position.y);
                if (dx || dy)
                    onMousePositionChanged(dx, dy);
            }
//            else {
//                switch (wParam) {
//                    case MK_RBUTTON:
//                        break;
//                    case MK_MBUTTON:
//                        break;
//                    default:
//                        break;
//                }
//            }


        case WM_PAINT: {
            SetDIBitsToDevice(
                    device_context, 0, 0,
                    frame_buffer.width,
                    frame_buffer.height, 0, 0, 0,
                    frame_buffer.height,
                    frame_buffer.pixels,
                    &info,
                    DIB_RGB_COLORS);
            ValidateRgn(window, NULL);
        } break;

        case WM_SIZE: {
            resizeFrameBuffer();
            updateAndRender();

        } break;

        case WM_DESTROY: {
            app.is_running = FALSE;
            PostQuitMessage(0);
        } break;

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
    window_class.lpszClassName  = WINDOW_CLASS;
    window_class.hInstance      = hInstance;
    window_class.lpfnWndProc    = WndProc;
    window_class.style          = CS_OWNDC|CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS;
    window_class.hbrBackground  = (HBRUSH)COLOR_WINDOW+1;
    window_class.hCursor        = LoadCursorA(NULL, IDC_ARROW);
//    window_class.hbrBackground  = NULL;
//    window_class.hCursor        = NULL;
    window_class.hIcon          = NULL;
    window_class.lpszMenuName   = NULL;
    window_class.cbClsExtra     = 0;
    window_class.cbWndExtra     = 0;

    RegisterClassA(&window_class);

    window = CreateWindowA(
            WINDOW_CLASS,
            TITLE,
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            INITIAL_WIDTH,
            INITIAL_HEIGHT,
            NULL,
            NULL,
            hInstance,
            NULL
    );
    if (!window) {
        return -1;
    }
    device_context = GetDC(window);  //GetDCEx(window, NULL, DCX_WINDOW);

//    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    ShowWindow(window, nCmdShow);
//    resizeFrameBuffer();

    UpdateWindow(window);

    while (app.is_running) {
        processMessages();
        updateAndRender();
    }

    return 0;
}

//LRESULT CALLBACK WndProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
//    switch (message) {
//        case WM_DESTROY: {
//            app.is_running = FALSE;
//            PostQuitMessage(0);
//
//        } break;
//
//        case WM_PAINT: {
//            PAINTSTRUCT ps;
//            HDC device_context = BeginPaint(window, &ps);
//            blit(device_context);
//            EndPaint(window, &ps);
//        } break;
//
//        case WM_SIZE: {
//            resizeFrameBuffer(window);
//            updateAndRender(window);
//
//        } break;
//
//        default:
//            return DefWindowProc(window, message, wParam, lParam);
//    }
//
//    return 0;
//}
//
//int CALLBACK WinMain(HINSTANCE instance, HINSTANCE prev_instance, LPSTR command_line, int show_code) {
//    LARGE_INTEGER performance_frequency;
//    QueryPerformanceFrequency(&performance_frequency);
//    counts_per_second = (f64)performance_frequency.QuadPart;
//
//    // Initialize the memory:
//    memory.address = (u8*)VirtualAlloc((LPVOID)memory.base, memory.size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
//    if (!memory.address)
//        return -1;
//
//    init_core();
//    init_renderer();
//
//    info.bmiHeader.biSize = sizeof(info.bmiHeader);
//    info.bmiHeader.biPlanes = 1;
//    info.bmiHeader.biBitCount = 32;
//    info.bmiHeader.biCompression = BI_RGB;
//
//    // Initialize the window and it's class:
//    WNDCLASSEXA window_class;
//    window_class.lpszClassName = "RenderEngineClass";
//    window_class.hIcon = LoadIcon(NULL, IDI_APPLICATION);
//    window_class.cbSize = sizeof(WNDCLASSEXA);
//    window_class.style = CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS;
//    window_class.lpfnWndProc = WndProc;
//    window_class.cbClsExtra = 0;
//    window_class.cbWndExtra = 0;
//    window_class.hInstance = instance;
//    window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
//    window_class.hbrBackground =(HBRUSH)COLOR_WINDOW;
//    window_class.lpszMenuName = 0;
//    window_class.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
//
//    if (!RegisterClassExA(&window_class))
//        return -1;
//
//    AdjustWindowRectEx(&rect, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_OVERLAPPEDWINDOW);
//    HWND window = CreateWindowExA(
//        0,
//        window_class.lpszClassName,
//        TITLE,
//        WS_OVERLAPPEDWINDOW|WS_VISIBLE,
//        CW_USEDEFAULT,
//        CW_USEDEFAULT,
//        rect.right - rect.left,
//        rect.bottom - rect.top,
//        NULL,
//        NULL,
//        instance,
//        NULL
//    );
//
//    if (!window)
//        return -1;
//
//    while (app.is_running) {
//        processMessages(window);
//        updateAndRender(window);
//    }
//
//    return 0;
//}