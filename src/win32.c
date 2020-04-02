#define PLATFORM_IS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "lib/core/perf.h"
#include "lib/core/hud.h"
#include "lib/engine.h"

#define RAW_INPUT_MAX_SIZE Kilobytes(1)
#define MEMORY_SIZE Gigabytes(1)
#define MEMORY_BASE Terabytes(2)

static bool is_running = true;
static f32 dx, dy;

static WNDCLASSA window_class;
static HWND window;
static HDC win_dc;
static BITMAPINFO info;
static RECT win_rect;
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

    OnFrameBufferResized();
}

void updateFrame() {
    OnFrameUpdate();
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
            updateFrame();
            break;

        case WM_PAINT:
            SetDIBitsToDevice(win_dc,
                    0, 0, frame_buffer.width, frame_buffer.height,
                    0, 0, 0, frame_buffer.height,
                    frame_buffer.pixels, &info, DIB_RGB_COLORS);

            ValidateRgn(window, NULL);

            break;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            switch ((u32)wParam) {
                case 'W': input.keyboard.pressed |= input.buttons.FORWARD; break;
                case 'A': input.keyboard.pressed |= input.buttons.LEFT; break;
                case 'S': input.keyboard.pressed |= input.buttons.BACKWARD; break;
                case 'D': input.keyboard.pressed |= input.buttons.RIGHT; break;
                case 'R': input.keyboard.pressed |= input.buttons.UP; break;
                case 'F': input.keyboard.pressed |= input.buttons.DOWN; break;

                case VK_SPACE:
                    if (input.mouse.is_captured) {
                        ReleaseCapture();
                        ShowCursor(true);
                        OnMouseCaptureChanged(false);
                    } else {
                        SetCapture(window);
                        ShowCursor(false);
                        OnMouseCaptureChanged(true);
                    }
                    break;

                case VK_TAB:
                    OnTabPressed();
                    break;

                case VK_ESCAPE:
                    is_running = false;
                    break;
            }
            break;

        case WM_SYSKEYUP:
        case WM_KEYUP:
            switch ((u32)wParam) {
                case 'W': input.keyboard.pressed &= (u8)~input.buttons.FORWARD; break;
                case 'A': input.keyboard.pressed &= (u8)~input.buttons.LEFT; break;
                case 'S': input.keyboard.pressed &= (u8)~input.buttons.BACKWARD; break;
                case 'D': input.keyboard.pressed &= (u8)~input.buttons.RIGHT; break;
                case 'R': input.keyboard.pressed &= (u8)~input.buttons.UP; break;
                case 'F': input.keyboard.pressed &= (u8)~input.buttons.DOWN; break;
            }
            break;

        case WM_MOUSEWHEEL:
            OnMouseWheelChanged(GET_WHEEL_DELTA_WPARAM(wParam) / 120.0f);
            break;

        case WM_INPUT:
            size_ri = 0;
            if (!GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &size_ri, size_rih) && size_ri &&
                 GetRawInputData((HRAWINPUT)lParam, RID_INPUT, raw_inputs, &size_ri, size_rih) == size_ri &&
                 raw_inputs->header.dwType == RIM_TYPEMOUSE) {
                raw_mouse = raw_inputs->data.mouse;

                if (     raw_mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN) input.mouse.pressed |= input.buttons.LEFT;
                else if (raw_mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP) input.mouse.pressed &= (u8)~input.buttons.LEFT;;

                if (     raw_mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN) input.mouse.pressed |= input.buttons.RIGHT;
                else if (raw_mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP) input.mouse.pressed &= (u8)~input.buttons.RIGHT;;

                if (     raw_mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN) input.mouse.pressed |= input.buttons.MIDDLE;
                else if (raw_mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP) input.mouse.pressed &= (u8)~input.buttons.MIDDLE;;

                dx = (f32)raw_mouse.lLastX;
                dy = (f32)raw_mouse.lLastY;

                if (dx || dy)
                    OnMousePositionChanged(dx, dy);
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
            getEngineTitle(),
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

    while (is_running) {
        while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageA(&message);
        }

        updateFrame();
    }

    return (int)message.wParam;
}