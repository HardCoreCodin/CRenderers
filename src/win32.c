#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define GET_X_LPARAM(lp)                        ((int)(short)LOWORD(lp))
#define GET_Y_LPARAM(lp)                        ((int)(short)HIWORD(lp))

#include "lib/core/perf.h"
#include "lib/input/mouse.h"
#include "lib/input/keyboard.h"
#include "lib/engine.h"

#define RAW_INPUT_MAX_SIZE Kilobytes(1)

static WNDCLASSA window_class;
static HWND window;
static HDC win_dc;
static BITMAPINFO info;
static RECT win_rect;
static RAWINPUT raw_inputs;
static HRAWINPUT raw_input_handle;
static RAWINPUTDEVICE raw_input_device;
static UINT raw_input_size;
static PUINT raw_input_size_ptr = (PUINT)(&raw_input_size);
static UINT raw_input_header_size = sizeof(RAWINPUTHEADER);

static u64 Win32_ticksPerSecond;
static LARGE_INTEGER performance_counter;

void Win32_printDebugString(char* str) { OutputDebugStringA(str); }
void Win32_updateWindowTitle() { SetWindowTextA(window, getTitle()); }
u64 Win32_getTicks() {
    QueryPerformanceCounter(&performance_counter);
    return (u64)performance_counter.QuadPart;
}

inline UINT getRawInput(LPVOID data) {
    return GetRawInputData(raw_input_handle, RID_INPUT, data, raw_input_size_ptr, raw_input_header_size);
}
inline bool hasRawInput() {
    return getRawInput(0) == 0 && raw_input_size != 0;
}
inline bool hasRawMouseInput(LPARAM lParam) {
    raw_input_handle = (HRAWINPUT)(lParam);
    return (
        hasRawInput() &&
        getRawInput((LPVOID)&raw_inputs) == raw_input_size &&
        raw_inputs.header.dwType == RIM_TYPEMOUSE
    );
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_DESTROY:
            is_running = false;
            PostQuitMessage(0);
            break;

        case WM_SIZE:
            GetClientRect(window, &win_rect);

            info.bmiHeader.biWidth = win_rect.right - win_rect.left;
            info.bmiHeader.biHeight = win_rect.top - win_rect.bottom;

            resize((u16)info.bmiHeader.biWidth, (u16)-info.bmiHeader.biHeight);

            break;

        case WM_PAINT:
            SetDIBitsToDevice(win_dc,
                    0, 0, frame_buffer.width, frame_buffer.height,
                    0, 0, 0, frame_buffer.height,
                    (u32*)frame_buffer.pixels, &info, DIB_RGB_COLORS);

            ValidateRgn(window, NULL);
            break;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            keyChanged((u32)wParam, true);
            break;

        case WM_SYSKEYUP:
        case WM_KEYUP:
            keyChanged((u32)wParam, false);
            break;

        case WM_MBUTTONUP:
            setMouseButtonUp(  &middle_mouse_button, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
            ReleaseCapture();
            ShowCursor(true);
            break;

        case WM_MBUTTONDOWN:
            setMouseButtonDown(&middle_mouse_button, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
            SetCapture(window);
            ShowCursor(false);
            break;

        case WM_LBUTTONDOWN: setMouseButtonDown(&left_mouse_button,  GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)); break;
        case WM_LBUTTONUP  : setMouseButtonUp(  &left_mouse_button,  GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)); break;
        case WM_RBUTTONDOWN: setMouseButtonDown(&right_mouse_button, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)); break;
        case WM_RBUTTONUP:   setMouseButtonUp(  &right_mouse_button, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)); break;

        case WM_LBUTTONDBLCLK:
            mouse_double_clicked = true;
            if (mouse_is_captured) {
                mouse_is_captured = false;
                ReleaseCapture();
                ShowCursor(true);
            } else {
                mouse_is_captured = true;
                SetCapture(window);
                ShowCursor(false);
            }
            break;

        case WM_MOUSEWHEEL:
            setMouseWheel((f32)(GET_WHEEL_DELTA_WPARAM(wParam)) / (f32)(WHEEL_DELTA));
            break;

        case WM_MOUSEMOVE:
            setMousePosition(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
            break;

        case WM_INPUT:
            if ((hasRawMouseInput(lParam)) && (
                raw_inputs.data.mouse.lLastX != 0 ||
                raw_inputs.data.mouse.lLastY != 0))
                setMouseMovement(
                    raw_inputs.data.mouse.lLastX,
                    raw_inputs.data.mouse.lLastY
                );

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

    LARGE_INTEGER performance_frequency;
    QueryPerformanceFrequency(&performance_frequency);
    Win32_ticksPerSecond = (u64)performance_frequency.QuadPart;

    initEngine(
        Win32_updateWindowTitle,
        Win32_printDebugString,
        Win32_getTicks,
        Win32_ticksPerSecond
    );

    up_key       = 'R';
    down_key     = 'F';
    forward_key  = 'W';
    backward_key = 'S';
    left_key     = 'A';
    right_key    = 'D';
    turn_right_key = 'E';
    turn_left_key  = 'Q';
    space_key      = VK_SPACE;
    shift_key      = VK_SHIFT;
    ctrl_key       = VK_CONTROL;
    alt_key        = VK_MENU;
    toggle_HUD_key = VK_TAB;
    toggle_BVH_key = '1';
    toggle_SSB_key = '2';

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
            RAY_TRACER_TITLE,
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

    raw_input_device.usUsagePage = 0x01;
    raw_input_device.usUsage = 0x02;
    if (!RegisterRawInputDevices(&raw_input_device, 1, sizeof(raw_input_device)))
        return -1;

    win_dc = GetDC(window);
    ShowWindow(window, nCmdShow);

    MSG message;
    while (is_running) {
        while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageA(&message);
        }
        updateAndRender();
        InvalidateRgn(window, NULL, FALSE);
    }

    return (int)message.wParam;
}