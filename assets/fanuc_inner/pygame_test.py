import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))

running = True
while running:
    for event in pygame.event.get():
        print(event)  # 应该打印出所有事件

        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            print("Key pressed:", event.key)  # 确认键盘输入

pygame.quit()
