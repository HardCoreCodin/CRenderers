#include <math.h>
#include "core.h"
#include "math2D.h"
#include "core2D.h"
#include "draw2D.h"

#define MAX_WIDTH 3840
#define TILE_SIZE 64
#define MAP_NUM_ROWS 13
#define MAP_NUM_COLS 20
#define MINIMAP_SCALE_FACTOR 0.2

Pixel FULL = {0xFFFFFFFF};
Pixel EMPTY = {0xFF000000};
Pixel VERTICAL = {0xFFFFFFFF};
Pixel HORIZONTAL = {0xFFCCCCCC};
Pixel CEILING = {0xFF333333};
Pixel FLOOR = {0xFF777777};
Pixel RAY = {0xFF0000FF};

static char* TITLE = "RayCast";

const u8 map[MAP_NUM_ROWS][MAP_NUM_COLS] = {
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1},
    {1, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1},
    {1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 2, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
};

Vector2* source_ray_directions;
Vector2* source_ray_direction;
Vector2* ray_directions;
Vector2* ray_direction;
Vector2* ray_hit_positions;
Vector2* ray_hit_position;
Vector2* ray_origin_to_hit_position;
u8* vertical_hits;

void rotateRayDirections() {
    f32 m11 = camera.matrix.m11; f32 m21 = camera.matrix.m21;
    f32 m12 = camera.matrix.m12; f32 m22 = camera.matrix.m22;

    source_ray_direction = source_ray_directions;
    ray_direction = ray_directions;

    f32 x, y;
    for (u16 ray_index = 0; ray_index < frame_buffer.width; ray_index++) {
        x = source_ray_direction->x;
        y = source_ray_direction->y;

        ray_direction->x = x*m11 + y*m21;
        ray_direction->y = x*m12 + y*m22;

        source_ray_direction++;
        ray_direction++;
    }
}

void generateRayDirections() {
    ray_direction = source_ray_directions;

    f32 ray_direction_length = 0;
    f32 squared_focal_length = camera.focal_length * camera.focal_length;
    f32 one_over_width = 1.0f / frame_buffer.width;
    f32 x, x2;

    for (u16 i = 0; i < frame_buffer.width; i++) {
        x = ((i + 0.5f) * one_over_width) - 0.5f;
        x2 = x * x;

        ray_direction_length = sqrtf(x2 + squared_focal_length);

        ray_direction->x = x / ray_direction_length;
        ray_direction->y = camera.focal_length / ray_direction_length;
        ray_direction++;
    }

    ray_directions = ray_direction;
    rotateRayDirections();
}

inline void fillTile(Pixel color, u16 position_x, u16 position_y) {
    u16 x, y;
    u16 width = frame_buffer.width; 
    
    pixel = frame_buffer.pixels + width*position_y + position_x;

    for (y = 0; y < TILE_SIZE; y++) {
        for (x = 0; x < TILE_SIZE; x++)
            *(pixel + x) = color;

        pixel += width;
    }
}

void render() {
    // Draw 3D view:
    // =============
    ray_direction = ray_directions;
    ray_hit_position = ray_hit_positions;
    
    u16 x, y, i;
    u16 width = frame_buffer.width;
    u16 height = frame_buffer.height;
    u16 half_width = width >> 1;
    u16 half_height = height >> 1;
    u16 wall_strip_half_height, wall_top, wall_bottom;
    f32 projected_wall_height = TILE_SIZE * half_width * camera.focal_length * 0.5f;

    for (i = 0; i < width; i++) {
        sub2D(ray_hit_position, camera.position, ray_direction);
        wall_strip_half_height = (u16)(projected_wall_height / dot2D(ray_direction, forward)) >> 1;
        wall_top = wall_strip_half_height > half_height ? 0 : half_height - wall_strip_half_height;
        wall_bottom = half_height + wall_strip_half_height > height ? height : half_height + wall_strip_half_height;

        pixel = frame_buffer.pixels + i;

        // set the color of the ceiling
        for (y = 0; y < wall_top; y++) {
            *pixel = CEILING;
            pixel += width;
        }
            
        // render the wall from wallTopPixel to wallBottomPixel
        for (y = wall_top; y < wall_bottom; y++) {
            *pixel = vertical_hits[i] ? VERTICAL : HORIZONTAL;
            pixel += width;
        }
            
        // set the color of the floor
        for (y = wall_bottom; y < height; y++) {
            *pixel = FLOOR;
            pixel += width;
        }

        ray_direction++;
        ray_hit_position++;
    }
    /*
    // Draw min-map:
    // =============
    u16 tile_size = (u16)(TILE_SIZE * MINIMAP_SCALE_FACTOR);
    u32 hOffset;
    u32 vOffset;
    u32 vOffsetStep = width * tile_size;
    u32 hMax = MAP_NUM_COLS * tile_size;
    u32 vMax = MAP_NUM_ROWS * vOffsetStep;
    
    for (vOffset = 0; vOffset < vMax; vOffset += vOffsetStep) {
        x = 0;
        for (hOffset = 0; hOffset < hMax; hOffset += tile_size)
            drawRect(map[y][x++] ? FULL : EMPTY, tile_size, tile_size, vOffset + hOffset);
        y++;
    }
    
    // Draw rays:
    // ==========
    u32 origin_x = (u32)(MINIMAP_SCALE_FACTOR * camera.position->x);
    u32 origin_y = (u32)(MINIMAP_SCALE_FACTOR * camera.position->y);
    u32 hit_x;
    u32 hit_y;

    ray_hit_position = ray_hit_positions;
    for (i = 0; i < width; i++) {
        hit_x = (u32)(MINIMAP_SCALE_FACTOR * ray_hit_position->x);
        hit_y = (u32)(MINIMAP_SCALE_FACTOR * ray_hit_position->y);

        drawLine(RAY, origin_x, origin_y, hit_x, hit_y);
        
        ray_hit_position++;
    }

    // Draw player:
    // ============
    drawRect(FULL, 1, 1, width*origin_y + origin_x);
    drawLine(FULL, origin_x, origin_y, origin_x + (u32)(forward->x * 40), origin_y + (u32)(forward->y*40));
*/}

inline f32 squaredDistanceBetweenPoints(f32 x1, f32 y1, f32 x2, f32 y2) {
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}
u8 mapHasWallAt(f32 x, f32 y) {
    if (x < 0 || x > frame_buffer.width || y < 0 || y > frame_buffer.height) {
        return TRUE;
    }
    u32 mapGridIndexX = flr(x / TILE_SIZE);
    u32 mapGridIndexY = flr(y / TILE_SIZE);
    return map[mapGridIndexY][mapGridIndexX] != 0;
}
void update() {
    if (keyboard.pressed)
        on_key_pressed();

    ray_direction = ray_directions;
    ray_hit_position = ray_hit_positions;

    u16 width = frame_buffer.width;
    u16 height = frame_buffer.height;

    f32 pos_x = camera.position->x;
    f32 pos_y = camera.position->y;

    u8 isRayFacingDown;
    u8 isRayFacingRight;
    u8 isRayFacingUp;
    u8 isRayFacingLeft;
    u8 foundHorzWallHit;
    u8 horzWallContent;
    u8 foundVertWallHit;
    u8 vertWallContent;
    f32 xintercept, xstep, rd_x, horzHitDistance, horzWallHitX, vertWallHitX, nextHorzTouchX, nextVertTouchX, xToCheck,
        yintercept, ystep, rd_y, vertHitDistance, horzWallHitY, vertWallHitY, nextHorzTouchY, nextVertTouchY, yToCheck;

    for (u16 stripId = 0; stripId < width; stripId++) {
        rd_x = ray_direction->x;
        rd_y = ray_direction->y;

        isRayFacingDown = rd_y > 0;
        isRayFacingRight = rd_x > 0;
        isRayFacingUp = !isRayFacingDown;
        isRayFacingLeft = !isRayFacingRight;

        ///////////////////////////////////////////
        // HORIZONTAL RAY-GRID INTERSECTION CODE
        ///////////////////////////////////////////
        foundHorzWallHit = FALSE;
        horzWallHitX = 0;
        horzWallHitY = 0;
        horzWallContent = 0;

        // Find the y-coordinate of the closest horizontal grid intersection
        yintercept = (f32)flr(pos_y / TILE_SIZE) * TILE_SIZE;
        yintercept += isRayFacingDown ? TILE_SIZE : 0;

        // Find the x-coordinate of the closest horizontal grid intersection
        xintercept = pos_x + (yintercept - pos_y) * rd_x / rd_y;

        // Calculate the increment xstep and ystep
        xstep = TILE_SIZE * rd_x / rd_y;
        xstep *= (isRayFacingLeft && xstep > 0) ? -1 : 1;
        xstep *= (isRayFacingRight && xstep < 0) ? -1 : 1;

        ystep = TILE_SIZE;
        ystep *= isRayFacingUp ? -1 : 1;

        nextHorzTouchX = xintercept;
        nextHorzTouchY = yintercept;

        // Increment xstep and ystep until we find a wall
        while (nextHorzTouchX >= 0 && nextHorzTouchX <= width && nextHorzTouchY >= 0 && nextHorzTouchY <= height) {
            xToCheck = nextHorzTouchX;
            yToCheck = nextHorzTouchY + (isRayFacingUp ? -1 : 0);

            if (mapHasWallAt(xToCheck, yToCheck)) {
                // found a wall hit
                horzWallHitX = nextHorzTouchX;
                horzWallHitY = nextHorzTouchY;
                horzWallContent = map[flr(yToCheck / TILE_SIZE)][flr(xToCheck / TILE_SIZE)];
                foundHorzWallHit = TRUE;
                break;
            } else {
                nextHorzTouchX += xstep;
                nextHorzTouchY += ystep;
            }
        }

        ///////////////////////////////////////////
        // VERTICAL RAY-GRID INTERSECTION CODE
        ///////////////////////////////////////////
        foundVertWallHit = FALSE;
        vertWallHitX = 0;
        vertWallHitY = 0;
        vertWallContent = 0;

        // Find the x-coordinate of the closest vertical grid intersection
        xintercept = (f32)flr(pos_x / TILE_SIZE) * TILE_SIZE;
        xintercept += isRayFacingRight ? TILE_SIZE : 0;

        // Find the y-coordinate of the closest horizontal grid intersection
        yintercept = pos_y + (xintercept - pos_x) * rd_y / rd_x;

        // Calculate the increment xstep and ystep
        ystep = TILE_SIZE * rd_y / rd_x;
        ystep *= (isRayFacingUp && ystep > 0) ? -1 : 1;
        ystep *= (isRayFacingDown && ystep < 0) ? -1 : 1;

        xstep = TILE_SIZE;
        xstep *= isRayFacingLeft ? -1 : 1;

        nextVertTouchX = xintercept;
        nextVertTouchY = yintercept;

        // Increment xstep and ystep until we find a wall
        while (nextVertTouchX >= 0 && nextVertTouchX <= width && nextVertTouchY >= 0 && nextVertTouchY <= height) {
            xToCheck = nextVertTouchX + (isRayFacingLeft ? -1 : 0);
            yToCheck = nextVertTouchY;

            if (mapHasWallAt(xToCheck, yToCheck)) {
                // found a wall hit
                vertWallHitX = nextVertTouchX;
                vertWallHitY = nextVertTouchY;
                vertWallContent = map[flr(yToCheck / TILE_SIZE)][flr(xToCheck / TILE_SIZE)];
                foundVertWallHit = TRUE;
                break;
            } else {
                nextVertTouchX += xstep;
                nextVertTouchY += ystep;
            }
        }

        // Calculate both horizontal and vertical hit distances and choose the smallest one
        horzHitDistance = foundHorzWallHit ? squaredDistanceBetweenPoints(pos_x, pos_y, horzWallHitX, horzWallHitY) : INT_MAX;
        vertHitDistance = foundVertWallHit ? squaredDistanceBetweenPoints(pos_x, pos_y, vertWallHitX, vertWallHitY) : INT_MAX;

        if (vertHitDistance < horzHitDistance) {
            ray_hit_position->x = vertWallHitX;
            ray_hit_position->y = vertWallHitY;
            vertical_hits[stripId] = TRUE;
        } else {
            ray_hit_position->x = horzWallHitX;
            ray_hit_position->y = horzWallHitY;
            vertical_hits[stripId] = FALSE;
        }

        ray_direction++;
        ray_hit_position++;
    }
}

void on_resize() {
    generateRayDirections();
}

void on_mouse_wheel(float amount) {
    camera.focal_length += amount / 10.0f;
    generateRayDirections();
}

void on_mouse_move() {
    rotateCamera();
    rotateRayDirections();
}

void init_renderer() {
    init_math2D();
    init_core2D();
    
    vertical_hits = (u8*)allocate_memory(MAX_WIDTH);
    source_ray_directions = (Vector2*)allocate_memory(MAX_WIDTH * PIXEL_SIZE);
    ray_origin_to_hit_position = (Vector2*)allocate_memory(sizeof(Vector2));
    ray_directions = (Vector2*)allocate_memory(MAX_WIDTH * PIXEL_SIZE);
    ray_hit_positions = (Vector2*)allocate_memory(sizeof(Vector2));
    camera.position = (Vector2*)allocate_memory(sizeof(Vector2));
    camera.position->x = 5;
    camera.position->y = 5;
}