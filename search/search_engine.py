from dotsandboxes import DotsAndBoxesEngine

class BaseSearchEngine():
    def __init__(self):
        pass

    # ✅ attr 기반 일괄 변경
    def configure(self, **kwargs):
        need_reset = False
        for k, v in kwargs.items():
            setattr(self, k, v)

        return self
    
    def search(self, eng: DotsAndBoxesEngine, state):
        raise NotImplementedError


