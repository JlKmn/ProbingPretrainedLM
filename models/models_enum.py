from enum import Enum

class Models(Enum):
    LPB = 1
    LPR = 2
    LB = 3
    LPRB = 4
    MPB = 5

    def get_run_name(model):
        if model == Models.LPB.value:
            return "LinearProbeBert"
        elif model == Models.LPR.value:
            return "LinearProbeRandom"
        elif model == Models.LB.value:
            return "LinearBert"
        elif model == Models.LPRB.value:
            return "LinearProbeResettedBert"
        elif model == Models.MPB.value:
            return "MultilayerProbeBert"