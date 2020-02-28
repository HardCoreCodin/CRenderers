#include <windows.h>
#include "renderer.h"

#ifdef PERF
#define DEBUG(work, message, multiplier) QueryPerformanceCounter(&before);work;QueryPerformanceCounter(&after);print_to_string(message,(u32)((f64)(after.QuadPart-before.QuadPart)*multiplier),message_string);OutputDebugStringA(message_string); 
#else
#define DEBUG(work, message) work;
#endif // PERF 

static char* CLASS = "RenderEngineClass";
static char title_string[32];
static LARGE_INTEGER before, after;
static u64 duration;
static f64 milliseconds_multiplier;
static f64 microseconds_multiplier;

static BITMAPINFO info;
//static PAINTSTRUCT paint;
POINTS point;
f32 dpi_scale_x;
f32 dpi_scale_y;

static MSG message;
static HDC device_context;
static HWND window;
static RECT rect = {0, 0, INITIAL_WIDTH, INITIAL_HEIGHT};


//typedef struct Win32DLL {
//    HMODULE dll;
//    FILETIME latest_write_time;
//    render *_render;
//    b32 is_valid;
//} Win32DLL;


inline void init_frame_buffer() {
    GetClientRect(window, &rect);
    frame_buffer.width = (u16)(rect.right - rect.left);
    frame_buffer.height = (u16)(rect.bottom - rect.top);
    frame_buffer.size = frame_buffer.width * frame_buffer.height;
    frame_buffer.pitch = frame_buffer.width * PIXEL_SIZE;
    info.bmiHeader.biWidth = frame_buffer.width;
    info.bmiHeader.biHeight = frame_buffer.height;
}

void update_and_render() {
    update();
    QueryPerformanceCounter(&before);
    render();
    QueryPerformanceCounter(&after);
    duration = after.QuadPart - before.QuadPart;
    print_title_to_string(frame_buffer.width, frame_buffer.height, (u16)(1000.0/(duration*milliseconds_multiplier)), TITLE, title_string);
    SetWindowTextA(window, title_string);
}

LRESULT CALLBACK windowCallback(HWND wnd, UINT msg, WPARAM WParam, LPARAM LParam) {
    switch(msg) {
        case WM_SIZE: 
            init_frame_buffer(); 
            on_resize();
            update_and_render();
            return 0;

        case WM_QUIT: 
            app.should_quit = 1; 
            return 0;

        case WM_KEYDOWN:
            switch ((u32)message.wParam) {
                case 'W': keyboard.pressed |= FORWARD; break;
                case 'A': keyboard.pressed |= LEFT; break;
                case 'S': keyboard.pressed |= BACKWARD; break;
                case 'D': keyboard.pressed |= RIGHT; break;
                case 'R': keyboard.pressed |= UP; break;
                case 'F': keyboard.pressed |= DOWN; break;

                case VK_ESCAPE: 
                    app.should_quit = 1; 
                    break;
            }
            return 0;

        case WM_KEYUP:
            switch ((u32)message.wParam) {
                case 'W': keyboard.pressed &= ~FORWARD; break;
                case 'A': keyboard.pressed &= ~LEFT; break;
                case 'S': keyboard.pressed &= ~BACKWARD; break;
                case 'D': keyboard.pressed &= ~RIGHT; break;
                case 'R': keyboard.pressed &= ~UP; break;
                case 'F': keyboard.pressed &= ~DOWN; break;
            }
            return 0;

        case WM_LBUTTONDBLCLK:
            if (app.is_active) {
                app.is_active = 0;
                mouse.prior_position.x = -1;
                ReleaseCapture();
            } else {
                app.is_active = 1;
                SetCapture(window);
            }
            return 0;

        case WM_MOUSEWHEEL:
            on_mouse_wheel(GET_WHEEL_DELTA_WPARAM(message.wParam) / 120.0f);
            return 0;

        case WM_MOUSEMOVE:
            if (app.is_active) {
                point = MAKEPOINTS(message.lParam);
                if (mouse.prior_position.x == -1) {
                    mouse.prior_position.x = point.x * dpi_scale_x;
                    mouse.prior_position.y = point.y * dpi_scale_y;
                } else {
                    mouse.current_position.x = point.x * dpi_scale_x;
                    mouse.current_position.y = point.y * dpi_scale_y;

                    on_mouse_move();

                    mouse.prior_position = mouse.current_position;
                }
            }
            return 0;

        case WM_PAINT: 
            //StretchDIBits(device_context, 0, 0, width, height, 0, 0, width, height, pixels, &info, DIB_RGB_COLORS, SRCCOPY);
            SetDIBitsToDevice(device_context, 0, 0, frame_buffer.width, frame_buffer.height, 0, 0, 0, frame_buffer.height, frame_buffer.pixels, &info, DIB_RGB_COLORS);
            //DEBUG(SetDIBitsToDevice(device_context, 0, 0, frame_buffer.width, frame_buffer.height, 0, 0, 0, frame_buffer.height, frame_buffer.pixels, &info, DIB_RGB_COLORS), "Blit:");
            return 0;

        case WM_ACTIVATEAPP: 
            return 0;
        
        //case WM_CREATE: 
        //    SetWindowLongPtrA(window, GWLP_USERDATA, ((LPCREATESTRUCT)LParam)->lpCreateParams); 
        //    return 0;
        
        case WM_CLOSE:
        case WM_DESTROY: 
            app.should_quit = 1; 
            PostQuitMessage(0); 
            return 0;

        default: 
            return DefWindowProcA(wnd, msg, WParam, LParam);
    }
}

int CALLBACK WinMain(HINSTANCE instance, HINSTANCE prev_instance, LPSTR command_line, int show_code) {
    LARGE_INTEGER performance_frequency;
    QueryPerformanceFrequency(&performance_frequency);
    microseconds_multiplier = 1000000.0 / performance_frequency.QuadPart;
    milliseconds_multiplier = 1000.0 / performance_frequency.QuadPart;

    // Initialize the memory:
    memory.address = (u8*)VirtualAlloc((LPVOID)memory.base, memory.size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    if (!memory.address)
        return -1;

    init_math();
    init_core();
    init_scene();
    init_renderer();

    info.bmiHeader.biSize = sizeof(info.bmiHeader);
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;

    // Initialize the window and it's class:
    WNDCLASSEXA window_class;
    window_class.cbSize = sizeof(WNDCLASSEXA);
    window_class.style = CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS;
    window_class.lpfnWndProc = windowCallback;
    window_class.cbClsExtra = 0;
    window_class.cbWndExtra = 0;
    window_class.hInstance = instance;
    window_class.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    window_class.hCursor = LoadCursor(NULL, IDC_ARROW);
    window_class.hbrBackground =(HBRUSH)COLOR_WINDOW;
    window_class.lpszMenuName = 0;
    window_class.lpszClassName = CLASS;
    window_class.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

    if (!RegisterClassExA(&window_class))
        return -1;

	AdjustWindowRectEx(&rect, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_OVERLAPPEDWINDOW);
	window = CreateWindowExA(
		WS_EX_OVERLAPPEDWINDOW,
		CLASS, 
        TITLE,
		WS_OVERLAPPEDWINDOW|WS_VISIBLE,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		INITIAL_WIDTH,
		INITIAL_HEIGHT, 
        NULL, 
        NULL,
        instance, 
        NULL
    );
    if (!window) 
        return -1;

    // Initialize the device context:
    device_context = GetDC(window);
    GetClientRect(window, &rect);

    dpi_scale_x = 96.0f / GetDeviceCaps(device_context, LOGPIXELSX);
    dpi_scale_y = 96.0f / GetDeviceCaps(device_context, LOGPIXELSY);

    
    while (!app.should_quit) {
        ULONGLONG currentTick = GetTickCount64();
        ULONGLONG endTick = currentTick + 1000/60;

        while (currentTick < endTick) {
            if (PeekMessageA(&message, window, 0, 0, PM_REMOVE)) {
                TranslateMessage(&message);
                DispatchMessageA(&message);
                currentTick = GetTickCount64();
            } else
                break;
        }
        update_and_render();
        //DEBUG(update_and_render(), "Render (ms):", milliseconds_multiplier);
    }

    //while (PeekMessage(&message, 0, 0, 0, PM_REMOVE)) {
    //    switch (message.message) {
    //        case WM_QUIT: 
    //            app.should_quit = 1; 
    //            break;
    //            
    //        case WM_KEYDOWN:
    //            switch ((u32)message.wParam) {
    //                case 'W': keyboard.pressed |= FORWARD; break;
    //                case 'A': keyboard.pressed |= LEFT; break;
    //                case 'S': keyboard.pressed |= BACKWARD; break;
    //                case 'D': keyboard.pressed |= RIGHT; break;
    //                case 'R': keyboard.pressed |= UP; break;
    //                case 'F': keyboard.pressed |= DOWN; break;

    //                case VK_ESCAPE: 
    //                    app.should_quit = 1; 
    //                    break;
    //            }
    //            break;

    //        case WM_KEYUP:
    //            switch ((u32)message.wParam) {
    //                case 'W': keyboard.pressed &= ~FORWARD; break;
    //                case 'A': keyboard.pressed &= ~LEFT; break;
    //                case 'S': keyboard.pressed &= ~BACKWARD; break;
    //                case 'D': keyboard.pressed &= ~RIGHT; break;
    //                case 'R': keyboard.pressed &= ~UP; break;
    //                case 'F': keyboard.pressed &= ~DOWN; break;
    //            }
    //            break;

    //        case WM_LBUTTONDBLCLK:
    //            if (app.is_active) {
    //                app.is_active = 0;
    //                ReleaseCapture();
    //            } else {
    //                app.is_active = 1;
    //                SetCapture(window);
    //            }

    //            break;

    //        case WM_MOUSEWHEEL:
    //            on_mouse_wheel(GET_WHEEL_DELTA_WPARAM(message.wParam) / 120.0f);
    //            break;

    //        case WM_MOUSEMOVE:
    //            if (app.is_active) {
    //                point = MAKEPOINTS(message.lParam);
    //                if (mouse.prior_position.x == -1) {
    //                    mouse.prior_position.x = point.x * dpi_scale_x;
    //                    mouse.prior_position.y = point.y * dpi_scale_y;
    //                } else {
    //                    mouse.current_position.x = point.x * dpi_scale_x;
    //                    mouse.current_position.y = point.y * dpi_scale_y;

    //                    on_mouse_move();

    //                    mouse.prior_position = mouse.current_position;
    //                }
    //            }
    //            break;

    //        default:
    //            TranslateMessage(&message);
    //            DispatchMessageA(&message);
    //    }

    //    if (app.should_quit) 
    //        break;
    //}
    
    return 0;
}
