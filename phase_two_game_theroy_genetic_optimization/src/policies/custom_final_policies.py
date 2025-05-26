import copy
import numpy as np

# Manual copy of policy so no generation from EvolutionnarySearch is needed


def cumul_exponential_incentive_policy(
    self,
    actions: np.ndarray,
    actors_priority: np.ndarray,
    avg_incomes: np.ndarray,
    water_pump: np.ndarray,
    avg_pump: np.ndarray,
    is_crisis: np.ndarray,
    water_flows: np.ndarray,
    quota: np.ndarray,
    DOE=15,
    DCR=10,
) -> np.ndarray:

    class CrisisLevel:
        NORMAL = -1.0
        ALERT = 0.0
        CRISIS = 1.0
        EXTREME_CRISIS = 2.0

    # This make the code bigger but also limit repetition problems
    if DOE == 0.1 and DCR == 0.05:
        # Hard coded params to avoid any problems
        # Scenario 0 optim results
        params = {
            "PF": 0.923169746128622,
            "PG": 3.134280569167306,
            "WF_EF": 0.5249153269703366,
            "PUR": 1.3654859762448144,
            "SUB_BC": 1.2762318960146912,
            "FIN_BC": 0.23665103851503674,
            "WF_SF": 2.929036282337239,
            "ANT_C_F": 0.07719003050635477,
            "ANT_N_F": 0.10005576673596114,
            "CF": 1.7691822513316218,
            "CG": 0.1902360535111736,
        }

        fine = np.zeros(self.nb_actors)
        crisis_level = is_crisis[-1]  # Current crisis level

        # If average income is negative, replace it with 0
        avg_incomes = np.where(avg_incomes < 0, 0, avg_incomes)

        # Defining a custom treshold that represent an anticipation crisis point based on DCR multiple
        critical_overall_demand_treshold = params["WF_SF"] * DCR

        # Copy to avoid reference use
        priority_factor = copy.copy(actors_priority)

        priority_factor[priority_factor == 2] = params["PF"]
        priority_factor[priority_factor == 1] = params["PF"] - 1.8 ** params["PG"]
        priority_factor[priority_factor == 0] = params["PF"] - 2.5 ** params["PG"]

        # As the simulation penalizes less subsidies we push more on it than fines
        SUBSIDY = -(
            # We make 5 times bigger subsidy as the simulation penalize 5 times less
            # So our politics align with the game rules and prefer pushing coop through subsidies rather than fines
            (DCR * params["SUB_BC"])
            * (1 / priority_factor)
        )

        FINE = (avg_incomes * params["FIN_BC"]) * (1 / priority_factor)

        exceding_quota_idx = water_pump > quota
        respecting_quota_idx = water_pump <= quota

        if (
            water_flows[-1] < critical_overall_demand_treshold
            and crisis_level == CrisisLevel.NORMAL
        ):
            # We anticipate a near crisis situation so we start applying fines
            # But it won't be as strong as in a real crisis

            fine[exceding_quota_idx] = FINE[exceding_quota_idx] * params["ANT_C_F"]
            fine[respecting_quota_idx] = (
                SUBSIDY[respecting_quota_idx] * params["ANT_C_F"]
            )
        else:

            match crisis_level:
                case CrisisLevel.NORMAL:
                    fine[exceding_quota_idx] = (
                        FINE[exceding_quota_idx] * params["ANT_N_F"]
                    )
                    fine[respecting_quota_idx] = (
                        SUBSIDY[respecting_quota_idx] * params["ANT_N_F"]
                    )
                    return fine

                case (
                    CrisisLevel.ALERT | CrisisLevel.CRISIS | CrisisLevel.EXTREME_CRISIS
                ):

                    if crisis_level == CrisisLevel.ALERT:
                        crisis_factor = params["CF"]
                    elif crisis_level == CrisisLevel.CRISIS:
                        crisis_factor = params["CF"] + params["CG"]
                    elif crisis_level == CrisisLevel.EXTREME_CRISIS:
                        crisis_factor = params["CF"] + 2 * params["CG"]
                    else:
                        print("WTF happening ?")
                        print("crisis_level", crisis_level)

                    actors_priority_below_crisis = actors_priority < crisis_level
                    actors_priority_above_crisis = actors_priority >= crisis_level

                    # Defectors
                    # Below priority and exceeding
                    actors_exceding_and_below_priority_idx = (
                        actors_priority_below_crisis == exceding_quota_idx
                    )
                    fine[actors_exceding_and_below_priority_idx] = (
                        FINE[actors_exceding_and_below_priority_idx]
                        * params["PF"]
                        * crisis_factor
                    )
                    # Above priority and exceeding
                    actors_exceding_and_above_priority_idx = (
                        actors_priority_above_crisis == exceding_quota_idx
                    )
                    fine[actors_exceding_and_above_priority_idx] = FINE[
                        actors_exceding_and_above_priority_idx
                    ]

                    # Cooperators
                    # Below priority and respecting
                    actors_respecting_and_below_priority_idx = (
                        actors_priority_below_crisis == respecting_quota_idx
                    )
                    fine[actors_respecting_and_below_priority_idx] = (
                        SUBSIDY[actors_respecting_and_below_priority_idx]
                        * params["PF"]
                        * crisis_factor
                    )
                    # Above priority and respecting
                    actors_respecting_and_above_priority_idx = (
                        actors_priority_above_crisis == respecting_quota_idx
                    )
                    fine[actors_respecting_and_above_priority_idx] = SUBSIDY[
                        actors_respecting_and_above_priority_idx
                    ]

                    return fine
        return fine

    else:
        # Scenario 1 optim results
        params = {
            "PF": 0.7266145696974551,
            "PG": 2.6016120263569538,
            "WF_EF": 0.865856531031441,
            "PUR": 0.9356609489016493,
            "SUB_BC": 3.2893318082328054,
            "FIN_BC": 0.060952784906565874,
            "WF_SF": 6.542628748726632,
            "ANT_C_F": 0.3005462481260572,
            "ANT_N_F": 0.0505896421462025,
            "CF": 1.519040171862454,
            "CG": 0.10856140246914334,
        }

        fine = np.zeros(self.nb_actors)
        crisis_level = is_crisis[-1]  # Current crisis level

        # If average income is negative, replace it with 0
        avg_incomes = np.where(avg_incomes < 0, 0, avg_incomes)

        # Defining a custom treshold that represent an anticipation crisis point based on DCR multiple
        critical_overall_demand_treshold = params["WF_SF"] * DCR

        # Copy to avoid reference use
        priority_factor = copy.copy(actors_priority)

        priority_factor[priority_factor == 2] = params["PF"]
        priority_factor[priority_factor == 1] = params["PF"] - 1.8 ** params["PG"]
        priority_factor[priority_factor == 0] = params["PF"] - 2.5 ** params["PG"]

        # As the simulation penalizes less subsidies we push more on it than fines
        SUBSIDY = -(
            # We make 5 times bigger subsidy as the simulation penalize 5 times less
            # So our politics align with the game rules and prefer pushing coop through subsidies rather than fines
            (DCR * params["SUB_BC"])
            * (1 / priority_factor)
        )

        FINE = (avg_incomes * params["FIN_BC"]) * (1 / priority_factor)

        exceding_quota_idx = water_pump > quota
        respecting_quota_idx = water_pump <= quota

        if (
            water_flows[-1] < critical_overall_demand_treshold
            and crisis_level == CrisisLevel.NORMAL
        ):
            # We anticipate a near crisis situation so we start applying fines
            # But it won't be as strong as in a real crisis

            fine[exceding_quota_idx] = FINE[exceding_quota_idx] * params["ANT_C_F"]
            fine[respecting_quota_idx] = (
                SUBSIDY[respecting_quota_idx] * params["ANT_C_F"]
            )
        else:

            match crisis_level:
                case CrisisLevel.NORMAL:
                    fine[exceding_quota_idx] = (
                        FINE[exceding_quota_idx] * params["ANT_N_F"]
                    )
                    fine[respecting_quota_idx] = (
                        SUBSIDY[respecting_quota_idx] * params["ANT_N_F"]
                    )
                    return fine

                case (
                    CrisisLevel.ALERT | CrisisLevel.CRISIS | CrisisLevel.EXTREME_CRISIS
                ):

                    if crisis_level == CrisisLevel.ALERT:
                        crisis_factor = params["CF"]
                    elif crisis_level == CrisisLevel.CRISIS:
                        crisis_factor = params["CF"] + params["CG"]
                    elif crisis_level == CrisisLevel.EXTREME_CRISIS:
                        crisis_factor = params["CF"] + 2 * params["CG"]
                    else:
                        print("WTF happening ?")
                        print("crisis_level", crisis_level)

                    actors_priority_below_crisis = actors_priority < crisis_level
                    actors_priority_above_crisis = actors_priority >= crisis_level

                    # Defectors
                    # Below priority and exceeding
                    actors_exceding_and_below_priority_idx = (
                        actors_priority_below_crisis == exceding_quota_idx
                    )
                    fine[actors_exceding_and_below_priority_idx] = (
                        FINE[actors_exceding_and_below_priority_idx]
                        * params["PF"]
                        * crisis_factor
                    )
                    # Above priority and exceeding
                    actors_exceding_and_above_priority_idx = (
                        actors_priority_above_crisis == exceding_quota_idx
                    )
                    fine[actors_exceding_and_above_priority_idx] = FINE[
                        actors_exceding_and_above_priority_idx
                    ]

                    # Cooperators
                    # Below priority and respecting
                    actors_respecting_and_below_priority_idx = (
                        actors_priority_below_crisis == respecting_quota_idx
                    )
                    fine[actors_respecting_and_below_priority_idx] = (
                        SUBSIDY[actors_respecting_and_below_priority_idx]
                        * params["PF"]
                        * crisis_factor
                    )
                    # Above priority and respecting
                    actors_respecting_and_above_priority_idx = (
                        actors_priority_above_crisis == respecting_quota_idx
                    )
                    fine[actors_respecting_and_above_priority_idx] = SUBSIDY[
                        actors_respecting_and_above_priority_idx
                    ]

                    return fine
        return fine


def cumul_exponential_quota(
    self,
    crisis_level: int,
    actors_priority: np.ndarray,
    avg_pump: np.ndarray,
    DOE: float,
    DCR: float,
) -> np.ndarray:

    class CrisisLevel:
        NORMAL = -1.0
        ALERT = 0.0
        CRISIS = 1.0
        EXTREME_CRISIS = 2.0

    # Manual copy of incentive variant
    # This make the code bigger but also limit repetition problems
    if DOE == 0.1 and DCR == 0.05:
        # Hard coded params to avoid any problems
        # Scenario 0 optim results
        params = {
            "PF": 0.923169746128622,
            "PG": 3.134280569167306,
            "WF_EF": 0.5249153269703366,
            "PUR": 1.3654859762448144,
            "SUB_BC": 1.2762318960146912,
            "FIN_BC": 0.23665103851503674,
            "WF_SF": 2.929036282337239,
            "ANT_C_F": 0.07719003050635477,
            "ANT_N_F": 0.10005576673596114,
            "CF": 1.7691822513316218,
            "CG": 0.1902360535111736,
        }

        # This will make higher priority actors having bigger quotas
        priority_factor = copy.copy(actors_priority)

        priority_factor[priority_factor == 2] = params["PF"]
        priority_factor[priority_factor == 1] = params["PF"] - 1.8 ** params["PG"]
        priority_factor[priority_factor == 0] = params["PF"] - 2.5 ** params["PG"]

        # print("crisis_level", crisis_level)
        if crisis_level == CrisisLevel.NORMAL:
            # Actors will be assigned quotas aligned with their priority
            # This will help us evaluate cooperation within incentive policy
            return avg_pump * priority_factor

        else:
            # Crisis times we estimate the available water and we distribute it according to the priority of the actors
            # Until it reach the ecological acceptable flow
            maximum_amount = 0.0

            match crisis_level:
                case CrisisLevel.ALERT:
                    maximum_amount = DOE
                case CrisisLevel.CRISIS:
                    maximum_amount = (DOE + DCR) / 2
                case CrisisLevel.EXTREME_CRISIS:
                    # When already bellow the ecological flow, we don't want anyone to pump
                    return np.zeros_like(avg_pump)

            estimated_flow = maximum_amount * params["WF_EF"]

            # Repartition by actor priority
            # Sort actors by priority, get indices
            sorted_indices = np.argsort(-actors_priority)
            quotas = np.zeros_like(avg_pump)

            for idx in sorted_indices:
                if estimated_flow <= DCR:
                    # If the remaining water is less than DCR, stop allocating, default value already 0
                    break
                pump = avg_pump[idx] * params["PUR"] * priority_factor[idx]
                if estimated_flow - pump >= DCR:
                    quotas[idx] = pump
                    estimated_flow -= pump
                else:
                    # If the remaining water is less than DCR the actor can't pump
                    quotas[idx] = 0

            return quotas
    else:
        # Scenario 1 optim results
        params = {
            "PF": 0.7266145696974551,
            "PG": 2.6016120263569538,
            "WF_EF": 0.865856531031441,
            "PUR": 0.9356609489016493,
            "SUB_BC": 3.2893318082328054,
            "FIN_BC": 0.060952784906565874,
            "WF_SF": 6.542628748726632,
            "ANT_C_F": 0.3005462481260572,
            "ANT_N_F": 0.0505896421462025,
            "CF": 1.519040171862454,
            "CG": 0.10856140246914334,
        }
        # This will make higher priority actors having bigger quotas
        priority_factor = copy.copy(actors_priority)

        priority_factor[priority_factor == 2] = params["PF"]
        priority_factor[priority_factor == 1] = params["PF"] - 1.8 ** params["PG"]
        priority_factor[priority_factor == 0] = params["PF"] - 2.5 ** params["PG"]

        # print("crisis_level", crisis_level)
        if crisis_level == CrisisLevel.NORMAL:
            # Actors will be assigned quotas aligned with their priority
            # This will help us evaluate cooperation within incentive policy
            return avg_pump * priority_factor

        else:
            # Crisis times we estimate the available water and we distribute it according to the priority of the actors
            # Until it reach the ecological acceptable flow
            maximum_amount = 0.0

            match crisis_level:
                case CrisisLevel.ALERT:
                    maximum_amount = DOE
                case CrisisLevel.CRISIS:
                    maximum_amount = (DOE + DCR) / 2
                case CrisisLevel.EXTREME_CRISIS:
                    # When already bellow the ecological flow, we don't want anyone to pump
                    return np.zeros_like(avg_pump)

            estimated_flow = maximum_amount * params["WF_EF"]

            # Repartition by actor priority
            # Sort actors by priority, get indices
            sorted_indices = np.argsort(-actors_priority)
            quotas = np.zeros_like(avg_pump)

            for idx in sorted_indices:
                if estimated_flow <= DCR:
                    # If the remaining water is less than DCR, stop allocating, default value already 0
                    break
                pump = avg_pump[idx] * params["PUR"] * priority_factor[idx]
                if estimated_flow - pump >= DCR:
                    quotas[idx] = pump
                    estimated_flow -= pump
                else:
                    # If the remaining water is less than DCR the actor can't pump
                    quotas[idx] = 0

            return quotas
