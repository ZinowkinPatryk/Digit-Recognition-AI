from PIL import Image

class ImageAnalysis:
    def __init__(self, path):
        self.image = Image.open(path).convert('L').resize((28, 28), Image.Resampling.LANCZOS)
        self.size = self.image.size
        self.pixelsValueTab = []

    def getPixels(self):
        for j in range(self.size[0]):
            for i in range(self.size[1]):
                self.pixelsValueTab.append(round(self.image.getpixel((i, j)) / 255, 2))
        return self.pixelsValueTab


if __name__ == "__main__":
    ImageAnalysis("./liczby/2.png").getPixels()


