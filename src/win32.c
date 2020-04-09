#define PLATFORM_IS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define GET_X_LPARAM(lp)                        ((int)(short)LOWORD(lp))
#define GET_Y_LPARAM(lp)                        ((int)(short)HIWORD(lp))

#include "lib/core/perf.h"
#include "lib/input/mouse.h"
#include "lib/input/keyboard.h"

#ifdef RAY_CASTER
#include "lib/engine2D.h"
#else
#include "lib/engine3D.h"
#endif

#define RAW_INPUT_MAX_SIZE Kilobytes(1)
#define MEMORY_SIZE Gigabytes(1)
#define MEMORY_BASE Terabytes(2)

static WNDCLASSA window_class;
static HWND window;
static HDC win_dc;
static BITMAPINFO info;
static RECT win_rect;
static RAWINPUTDEVICE rid;
static RAWINPUT* raw_inputs;
static RAWMOUSE raw_mouse;
static UINT size_ri, size_rih = sizeof(RAWINPUTHEADER);

static u64 ticks_of_last_frame, ticks_of_current_frame, target_ticks_per_frame;

#define RELEASE_MOUSE { \
    mouse.is_captured = false; \
    ReleaseCapture(); \
    ShowCursor(true); \
}
#define CAPTURE_MOUSE { \
    mouse.is_captured = true; \
    SetCapture(window); \
    ShowCursor(false); \
}

inline void resizeFrameBuffer() {
    GetClientRect(window, &win_rect);

    info.bmiHeader.biWidth = win_rect.right - win_rect.left;
    info.bmiHeader.biHeight = win_rect.top - win_rect.bottom;

    frame_buffer.width = (u16)info.bmiHeader.biWidth;
    frame_buffer.height = (u16)-info.bmiHeader.biHeight;
    frame_buffer.size = frame_buffer.width * frame_buffer.height;

    OnFrameBufferResized();
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_DESTROY:
            engine.is_running = false;
            PostQuitMessage(0);
            break;

        case WM_SIZE:
            resizeFrameBuffer();
            OnFrameUpdate();
            break;

        case WM_PAINT:
            OnFrameUpdate();

            GET_TICKS(ticks_of_current_frame);
            while (ticks_of_current_frame - ticks_of_last_frame < target_ticks_per_frame) {
                GET_TICKS(ticks_of_current_frame);
            }

            SetDIBitsToDevice(win_dc,
                    0, 0, frame_buffer.width, frame_buffer.height,
                    0, 0, 0, frame_buffer.height,
                    frame_buffer.pixels, &info, DIB_RGB_COLORS);

            ValidateRgn(window, NULL);

            GET_TICKS(ticks_of_last_frame);
            break;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            if ((u32)wParam == VK_ESCAPE)
                engine.is_running = false;
            else
                OnKeyDown((u32)wParam);
            break;

        case WM_SYSKEYUP:
        case WM_KEYUP:
            OnKeyUp((u32)wParam);
            break;

        case WM_LBUTTONDOWN:
            QueryPerformanceCounter(&perf_counter);
            OnMouseLeftButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (u64)perf_counter.QuadPart);
            break;

        case WM_RBUTTONDOWN:
            QueryPerformanceCounter(&perf_counter);
            OnMouseRightButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (u64)perf_counter.QuadPart);
            CAPTURE_MOUSE
            break;

        case WM_MBUTTONDOWN:
            QueryPerformanceCounter(&perf_counter);
            OnMouseMiddleButtonDown(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (u64)perf_counter.QuadPart);
            CAPTURE_MOUSE
            break;

        case WM_LBUTTONUP:
            QueryPerformanceCounter(&perf_counter);
            OnMouseLeftButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (u64)perf_counter.QuadPart);
            break;

        case WM_RBUTTONUP:
            QueryPerformanceCounter(&perf_counter);
            OnMouseRightButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (u64)perf_counter.QuadPart);
            RELEASE_MOUSE
            break;

        case WM_MBUTTONUP:
            QueryPerformanceCounter(&perf_counter);
            OnMouseMiddleButtonUp(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam), (u64)perf_counter.QuadPart);
            RELEASE_MOUSE
            break;

        case WM_LBUTTONDBLCLK:
            OnMouseDoubleClicked(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
            if (mouse.is_captured) RELEASE_MOUSE else CAPTURE_MOUSE
            break;

        case WM_MOUSEWHEEL:
            OnMouseWheelScrolled(GET_WHEEL_DELTA_WPARAM(wParam) / (float)WHEEL_DELTA);
            break;

        case WM_MOUSEMOVE:
            OnMouseMovedAbsolute(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
            break;

        case WM_INPUT:
            size_ri = 0;
            if (!GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &size_ri, size_rih) && size_ri &&
                 GetRawInputData((HRAWINPUT)lParam, RID_INPUT, raw_inputs, &size_ri, size_rih) == size_ri &&
                 raw_inputs->header.dwType == RIM_TYPEMOUSE) {
                raw_mouse = raw_inputs->data.mouse;
                if (raw_mouse.lLastX || raw_mouse.lLastY)
                    OnMouseMovedRelative((s16)raw_mouse.lLastX, (s16)raw_mouse.lLastY);
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
    // Initialize the memory:
    memory.address = (u8*)VirtualAlloc(
            (LPVOID)MEMORY_BASE,
            MEMORY_SIZE,
            MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    if (!memory.address)
        return -1;

    initRenderEngine();

    target_ticks_per_frame = perf.ticks_per_second / 60;

    buttons.up.key = 'R';
    buttons.down.key = 'F';
    buttons.left.key = 'A';
    buttons.right.key = 'D';
    buttons.forward.key = 'W';
    buttons.back.key = 'S';
    buttons.hud.key = VK_TAB;
    buttons.rat.key = VK_CONTROL;

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
            500,
            400,

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
    ShowWindow(window, nCmdShow);

    MSG message;
    GET_TICKS(ticks_of_last_frame);

    while (engine.is_running) {
        while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageA(&message);
        }
                    InvalidateRgn(window, NULL, FALSE);
    }

    return (int)message.wParam;
}