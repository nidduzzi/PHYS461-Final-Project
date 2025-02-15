class Solver:
    def __init__(self) -> None:
        y = []
        self.idx = 0

    def stepH(self, h, inplace=False) -> None:
        y1 = self.y + h*self.yy
        if inplace:
            self.setNext(y1)
        return y1

    def setNext(self, x) -> None:
        self.idx += 1
        self.y[self.idx] = x