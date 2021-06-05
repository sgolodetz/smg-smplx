import smplx

from smg.skeletons import Skeleton


class SMPLBody:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, gender: str, *, model_folder: str = "D:/smplx/models"):
        # noinspection PyTypeChecker
        self.__model: smplx.SMPL = smplx.create(model_folder, "smpl", gender=gender)

    # PUBLIC METHODS

    def render(self) -> None:
        pass

    def set_from_skeleton(self, skeleton: Skeleton) -> None:
        pass
