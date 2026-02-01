import pygame, random, imageio, numpy as np
from collections import deque

# ---------- CONFIG ----------
W, H = 25, 25
CELL = 20
FPS = 60
OUT = "maze_solution.mp4"
# ----------------------------

pygame.init()
screen = pygame.display.set_mode((W*CELL, H*CELL))
clock = pygame.time.Clock()

maze = [[1]*W for _ in range(H)]
dirs = [(2,0),(-2,0),(0,2),(0,-2)]

def carve(x,y):
    maze[y][x] = 0
    random.shuffle(dirs)
    for dx,dy in dirs:
        nx, ny = x+dx, y+dy
        if 0 < nx < W and 0 < ny < H and maze[ny][nx]:
            maze[y+dy//2][x+dx//2] = 0
            carve(nx, ny)

carve(1,1)
start, end = (1,1), (W-2,H-2)

def solve():
    q = deque([(start, [start])])
    seen = {start}
    while q:
        (x,y), path = q.popleft()
        if (x,y) == end: return path
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx,y+dy
            if 0<=nx<W and 0<=ny<H and not maze[ny][nx] and (nx,ny) not in seen:
                seen.add((nx,ny))
                q.append(((nx,ny), path+[(nx,ny)]))

path = solve()
writer = imageio.get_writer(OUT, fps=FPS)

def draw(p=None):
    screen.fill((0,0,0))
    for y in range(H):
        for x in range(W):
            if maze[y][x]:
                pygame.draw.rect(screen,(255,255,255),(x*CELL,y*CELL,CELL,CELL))
    for x,y in [start,end]:
        pygame.draw.rect(screen,(0,255,0),(x*CELL,y*CELL,CELL,CELL))
    if p:
        for x,y in p:
            pygame.draw.rect(screen,(255,0,0),(x*CELL,y*CELL,CELL,CELL))
    pygame.display.flip()

# ---------- RECORD ----------
for i in range(len(path)):
    for e in pygame.event.get():
        if e.type == pygame.QUIT: quit()
    draw(path[:i+1])
    frame = pygame.surfarray.array3d(screen)
    writer.append_data(np.transpose(frame,(1,0,2)))
    clock.tick(FPS)

writer.close()
pygame.quit()
print(f"Saved video → {OUT}")
