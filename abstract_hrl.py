import abc
import numpy


class GameState(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass


class GoalHrl(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def higher_level(self, s):
        pass

    def lower_level(self, s, g):
        pass


class FuRL(GoalHrl):
    def higher_level(self, s: GameState) -> (int, int):
        pass

    def lower_level(self, s: GameState, g: int) -> int:
        pass