from abc import abstractmethod, ABC


class Scheduler(ABC):
    @abstractmethod
    def get_betas(self):
        pass

    @abstractmethod
    def get_alphas(self):
        pass