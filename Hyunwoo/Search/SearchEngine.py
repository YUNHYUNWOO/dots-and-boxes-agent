from DotsAndBoxes import DotsAndBoxesEngine

class BaseSearchEngine():
    def __init__(self):
        pass

    def search(self, eng: DotsAndBoxesEngine, state):
        raise NotImplementedError

