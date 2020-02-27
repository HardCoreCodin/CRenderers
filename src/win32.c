#include <windows.h>
#include "renderer.c"

#define PERFORMANCE_RESULT_STRING_LENGTH 16
#define INITIAL_WIDTH 800
#define INITIAL_HEIGHT 600
#define PIXEL_SIZE 4

static char* CLASS = "Renderer";
static char* TITLE = "RendererClass";

static void* memory;
static u64 MAX_RENDER_TARGET_SIZE = Megabytes(8 * PIXEL_SIZE);
static u64 PERMANENT_MEMORY_SIZE = Megabytes(64);
static u64 MEMORY_SIZE = Gigabytes(1);
static u64 MEMORY_BASE = Terabytes(2);

static void* permanent_memory;
static void* transient_memory;

static u8 quit;
static u8* pixels; // BB GG RR XX
static u16 width;
static u16 height;
static u32 pitch;
static u32 pixel_count;
static size_t window_size;
static Keyboard keyboard;

static BITMAPINFO info;
static PAINTSTRUCT paint;
static MSG message;
static HDC device_context;
static HWND window;
static RECT rect = {0, 0, INITIAL_WIDTH, INITIAL_HEIGHT};

#ifdef PERF
static char performance_results[16];
static f64 microseconds_multiplier;
static LARGE_INTEGER before_rendering, after_rendering, after_blitting;
#endif // PERF

//typedef struct Win32DLL {
//    HMODULE dll;
//    FILETIME latest_write_time;
//    render *_render;
//    b32 is_valid;
//} Win32DLL;


inline void init_frame_buffer() {
    GetClientRect(window, &rect);
    width = rect.right - rect.left;
    height = rect.bottom - rect.top;
    pixel_count = width * height;
    pitch = width * PIXEL_SIZE;
    info.bmiHeader.biWidth = width;
    info.bmiHeader.biHeight = -height;
}

inline void update_frame_buffer() {
#ifdef PERF
    QueryPerformanceCounter(&before_rendering);
#endif // PERF
    render(width, height, pixels);
#ifdef PERF
    QueryPerformanceCounter(&after_rendering);
#endif // PERF 
    //StretchDIBits(device_context, 0, 0, width, height, 0, 0, width, height, pixels, &info, DIB_RGB_COLORS, SRCCOPY);
    SetDIBitsToDevice(device_context, 0, 0, width, height, 0, 0, 0, height, pixels, &info, DIB_RGB_COLORS);
#ifdef PERF
    QueryPerformanceCounter(&after_blitting);
	
    print_numbers_to_string(
        (u32)((f64)(after_rendering.QuadPart - before_rendering.QuadPart) * microseconds_multiplier),
        (u32)((f64)(after_blitting.QuadPart - after_rendering.QuadPart) * microseconds_multiplier),
        performance_results
    );
    OutputDebugStringA(performance_results);
#endif // PERF 
}

LRESULT CALLBACK windowCallback(HWND window, UINT message, WPARAM WParam, LPARAM LParam) {
    switch(message) {
        case WM_SIZE: init_frame_buffer(); return 0;
        case WM_PAINT: update_frame_buffer(); return 0;
        default: return DefWindowProcA(window, message, WParam, LParam);
        //case WM_ACTIVATEAPP: return 0;
        //case WM_CREATE: SetWindowLongPtrA(window, GWLP_USERDATA, ((LPCREATESTRUCT)LParam)->lpCreateParams); return 0;
        //case WM_CLOSE:
        //case WM_DESTROY: quit = 1; PostQuitMessage(0); return 0;
    }
}

int CALLBACK WinMain(HINSTANCE instance, HINSTANCE prev_instance, LPSTR command_line, int show_code) {
#ifdef PERF
    LARGE_INTEGER performance_frequency;
    QueryPerformanceFrequency(&performance_frequency);
    microseconds_multiplier = 1000000.0f / performance_frequency.QuadPart;
#endif // PERF

    info.bmiHeader.biSize = sizeof(info.bmiHeader);
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;

    // Initialize the window and it's class:
    WNDCLASSEXA window_class = {
        sizeof(WNDCLASSEXA),
        CS_HREDRAW|CS_VREDRAW,
        windowCallback, 0, 0, 
        instance,
        LoadIcon(NULL, IDI_APPLICATION),
        LoadCursor(NULL, IDC_ARROW),
        (HBRUSH)COLOR_WINDOW, 0,
        CLASS,
        LoadIcon(NULL, IDI_APPLICATION)
    };	
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
    
    // Initialize the device context and the frame buffer:
    device_context = GetDC(window);
    GetClientRect(window, &rect);
    init_frame_buffer();
                
    // Initialize the memory:
    memory = VirtualAlloc(MEMORY_BASE, MEMORY_SIZE, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    pixels = (u32*)memory; 
    permanent_memory = (u8*)memory + MAX_RENDER_TARGET_SIZE;
    transient_memory = (u8*)memory + PERMANENT_MEMORY_SIZE;

    // The engine loop:
    while (PeekMessage(&message, 0, 0, 0, PM_REMOVE)) {
        switch (message.message) {
            case WM_QUIT: 
                quit = 1; 
                break;
                
            case WM_KEYDOWN:
                switch ((u32)message.wParam) {
                    case 'W': keyboard |= FORWARD; break;
                    case 'A': keyboard |= LEFT; break;
                    case 'S': keyboard |= BACKWARD; break;
                    case 'D': keyboard |= RIGHT; break;
                    case 'R': keyboard |= UP; break;
                    case 'F': keyboard |= DOWN; break;

                    case VK_ESCAPE: 
                        quit = 1; 
                        break;
                }
                break;

            case WM_KEYUP:
                switch ((u32)message.wParam) {
                    case 'W': keyboard &= ~FORWARD; break;
                    case 'A': keyboard &= ~LEFT; break;
                    case 'S': keyboard &= ~BACKWARD; break;
                    case 'D': keyboard &= ~RIGHT; break;
                    case 'R': keyboard &= ~UP; break;
                    case 'F': keyboard &= ~DOWN; break;
                }
                break;

            default:
                TranslateMessage(&message);
                DispatchMessageA(&message);
        }

        if (quit) 
            break;
    }
    
    return 0;
}
