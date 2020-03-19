#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "ray_trace.h"

static char title_string[32];
static f64 counts_per_second, counts_per_millisecond;

static WNDCLASSA window_class;
static HWND window;
static HDC winDC, memDC, memDC2;
static HBITMAP bitmap;
static HBITMAP memory_bitmap;
static BITMAPINFO info;
static RECT rect;
static POINT current_mouse_position, prior_mouse_position;
static HFONT font;
static LARGE_INTEGER before_rendering, after_rendering;

inline void resizeFrameBuffer() {
//    if (memory_bitmap) DeleteObject(memory_bitmap);
//
//    GetClientRect(window, &rect);
//
//    info.bmiHeader.biWidth = rect.right - rect.left;
//    info.bmiHeader.biHeight = rect.bottom - rect.top;
//
////    memory_bitmap = CreateCompatibleBitmap(winDC, info.bmiHeader.biWidth, info.bmiHeader.biHeight);
////    SelectObject(memDC, memory_bitmap);
//
//    memory_bitmap = CreateDIBSection(
//            memDC,
//            &info,
//            DIB_RGB_COLORS,
//            (void*)&frame_buffer.pixels,
//            0,
//            0);
//    SelectObject(memDC, memory_bitmap);
//
//    frame_buffer.width = (u16)info.bmiHeader.biWidth;
//    frame_buffer.height = (u16)info.bmiHeader.biHeight;
//    frame_buffer.size = frame_buffer.width * frame_buffer.height;

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

//    QueryPerformanceCounter(&before_rendering);
    render();
//    QueryPerformanceCounter(&after_rendering);

    RedrawWindow(window, NULL, NULL, RDW_INVALIDATE|RDW_NOCHILDREN|RDW_UPDATENOW);
//    InvalidateRgn(window, 0, false);
//    UpdateWindow(window);
}
//BOOL DrawBitmap (HBITMAP hBitmap, DWORD dwROP)
//{
//    BITMAP    Bitmap;
//    BOOL      bResult;
//
//    GetObject(hBitmap, sizeof(BITMAP), (LPSTR)&Bitmap);
//    SelectObject(hDCBits, hBitmap);
//    bResult = BitBlt(hDC, 0, 0, Bitmap.bmWidth, Bitmap.bmHeight, hDCBits, 0, 0, dwROP);
//    DeleteDC(hDCBits);
//
//    return bResult;
//}
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_DESTROY:
            app.is_running = false;
            PostQuitMessage(0);
            break;

        case WM_ERASEBKGND:
            return 1;

        case WM_SIZE:
            resizeFrameBuffer();
            updateAndRender();
            break;

        case WM_PAINT:
            QueryPerformanceCounter(&before_rendering);


            // 9)
            GdiFlush();
            BitBlt(memDC2, 0, 0, frame_buffer.width, frame_buffer.height,
                   memDC, 0, 0, SRCCOPY);
            BitBlt(winDC, 0, 0, frame_buffer.width, frame_buffer.height,
                   memDC2, 0, 0, SRCCOPY);


//            SetDIBitsToDevice(
//                    winDC, 0, 0,
//                    frame_buffer.width,
//                    frame_buffer.height, 0, 0, 0,
//                    frame_buffer.height,
//                    frame_buffer.pixels,
//                    &info,
//                    DIB_RGB_COLORS);
////            RedrawWindow(window, NULL, NULL, RDW_VALIDATE|RDW_NOERASE);

            QueryPerformanceCounter(&after_rendering);
            printTitleIntoString(
                    frame_buffer.width,
                    frame_buffer.height,
                    (u16)(
                            (
                                    after_rendering.QuadPart - before_rendering.QuadPart
                            )),
                    "RT",
                    title_string);

//            ExtTextOutA(winDC, 10, 50, ETO_OPAQUE, NULL, title_string, 32, NULL);
//            SetWindowTextA(window, title_string);
            TextOutA(winDC, 10, 50, title_string, strlen(title_string));
            ValidateRgn(window, NULL);
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

int APIENTRY WinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPSTR     lpCmdLine,
                     int       nCmdShow) {
    LARGE_INTEGER performance_frequency;
    QueryPerformanceFrequency(&performance_frequency);
    counts_per_second = (f64)performance_frequency.QuadPart;
    counts_per_millisecond = (f64)performance_frequency.QuadPart / 1000;

    // Initialize the memory:
    memory.address = (u8*)VirtualAlloc((LPVOID)memory.base, memory.size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    if (!memory.address)
        return -1;

//    init_core();
    init_renderer();

    //    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, false);
    info.bmiHeader.biSize        = sizeof(info.bmiHeader);
    info.bmiHeader.biCompression = BI_RGB;
    info.bmiHeader.biBitCount    = 32;
    info.bmiHeader.biPlanes      = 1;

    window_class.lpszClassName  = "RnDer";
    window_class.hInstance      = hInstance;
    window_class.lpfnWndProc    = WndProc;
    window_class.style          = CS_OWNDC|CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS;
    window_class.hCursor        = LoadCursorA(0, IDC_ARROW);

    window_class.hbrBackground  = 0;
    window_class.hIcon          = 0;
    window_class.lpszMenuName   = 0;
    window_class.cbClsExtra     = 0;
    window_class.cbWndExtra     = 0;
//    window_class.hbrBackground  = (HBRUSH)COLOR_WINDOW+1;
//    window_class.hCursor        = NULL;

    RegisterClassA(&window_class);
//    CreateCompatibleBitmap
    window = CreateWindowA(
            window_class.lpszClassName,
            TITLE,
            WS_OVERLAPPEDWINDOW,

            CW_USEDEFAULT,
            CW_USEDEFAULT,
            INITIAL_WIDTH,
            INITIAL_HEIGHT,

            0,
            0,
            hInstance,
            0
    );
    if (!window)
        return -1;

    winDC = GetDC(window);  //GetDCEx(window, NULL, DCX_WINDOW);
    SetBkMode( winDC, TRANSPARENT );

    GetClientRect(window, &rect);

    info.bmiHeader.biWidth = rect.right - rect.left;
    info.bmiHeader.biHeight = rect.bottom - rect.top;

    frame_buffer.width = (u16)info.bmiHeader.biWidth;
    frame_buffer.height = (u16)info.bmiHeader.biHeight;
    frame_buffer.size = frame_buffer.width * frame_buffer.height;


    // 1)
    memory_bitmap = CreateDIBSection(
            winDC,
            &info,
            DIB_RGB_COLORS,
            (void**)&frame_buffer.pixels,
            0,
            0);
    // 2)
    memDC = CreateCompatibleDC(winDC);
    // 3)
    SelectObject(memDC, memory_bitmap);

    // 4)
    bitmap = CreateCompatibleBitmap(winDC, info.bmiHeader.biWidth, info.bmiHeader.biHeight);
    // 5)
    memDC2 = CreateCompatibleDC(winDC);
    // 6)
    SelectObject(memDC2, bitmap);

//    NONCLIENTMETRICS ncm = {0};
//    ncm.cbSize= sizeof(NONCLIENTMETRICS);
//    //Creates a font from the current theme's caption font
//    SystemParametersInfo(SPI_GETNONCLIENTMETRICS, NULL, &ncm, NULL);
//    font = CreateFontIndirect(&ncm.lfCaptionFont);

    font = GetStockObject(SYSTEM_FONT);
    SelectObject(winDC, font);
    SetTextColor(winDC, 0x0000FF00);

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