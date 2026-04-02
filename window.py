import sys
import pygame
from images import ImageAnalysis
from model import NeuralNet, Trainer


class WindowDraw:
    def __init__(self):
        self.window = pygame.display.set_mode((600, 600))
        self.model = NeuralNet(784, 128, 10)
        self.trainer = Trainer(self.model, lr=0.001)

    def windowRun(self):
        pygame.display.set_caption("Window")
        pygame.init()
        mouse = pygame.mouse.get_pos()
        self.window.fill((0, 0, 0))
        while True:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.window.fill((0, 0, 0))
                        pygame.display.flip()
                    if event.key == pygame.K_t:
                        self.saveDraw()
                        number = ImageAnalysis("windowScreenshot.png").getPixels()
                        self.trainer.numberWindow(number)
                if pygame.mouse.get_pressed()[0]:
                    pygame.draw.circle(self.window, (255, 255, 255), (mouse[0], mouse[1]), 25)
                    pygame.display.update()

    def saveDraw(self):
        pygame.image.save(self.window, "windowScreenshot.png")
