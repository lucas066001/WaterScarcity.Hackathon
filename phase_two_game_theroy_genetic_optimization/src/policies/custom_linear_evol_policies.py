import copy
import random
import numpy as np


class CrisisLevel:
    NORMAL = -1.0
    ALERT = 0.0
    CRISIS = 1.0
    EXTREME_CRISIS = 2.0


def tuned_make_linear_quota_function(self, params):
    """
    Based on given parameters it generate a quota policy.

    Returns:
        func: Quota policy.
    """

    def tuned_quota(
        self,
        crisis_level: int,
        actors_priority: np.ndarray,
        avg_pump: np.ndarray,
        DOE: float,
        DCR: float,
    ) -> np.ndarray:
        """
        Hard quota policy based on priority and crisis level.

        Sets quotas to zero for actors with priority less than the current crisis level.

        Args:
            crisis_level: Current water crisis level.
            actors_priority: Priority levels for each actor.
            avg_pump: Average pumping for each actor.
            DOE: Ecological optimal flow threshold.
            DCR: Crisis flow threshold.

        Returns:
            Array of water quotas for each actor.
        """

        # This will make higher priority actors having bigger quotas
        priority_factor = copy.copy(actors_priority)

        priority_factor[priority_factor == 2] = params["PF"]
        priority_factor[priority_factor == 1] = params["PF"] - params["PG"]
        priority_factor[priority_factor == 0] = params["PF"] - 2 * params["PG"]

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

    return tuned_quota


def tuned_make_linear_incentive_function(self, params):
    """
    Based on given parameters it generate an incentive policy.

    Returns:
        func: Incentive policy.
    """

    def tuned_incentive_policy(
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
        """
        Custom incentive policy that applies fines for exceeding quota and subsidies for cooperation.

        Returns an array of incentives (positive values = fines, negative values = subsidies)
        """
        fine = np.zeros(self.nb_actors)
        crisis_level = is_crisis[-1]  # Current crisis level

        # If average income is negative, replace it with 0
        avg_incomes = np.where(avg_incomes < 0, 0, avg_incomes)

        # Defining a custom treshold that represent an anticipation crisis point based on DCR multiple
        critical_overall_demand_treshold = params["WF_SF"] * DCR

        # This will make higher priority actors having bigger quotas
        priority_factor = copy.copy(actors_priority)

        priority_factor[priority_factor == 2] = params["PF"]
        priority_factor[priority_factor == 1] = params["PF"] - params["PG"]
        priority_factor[priority_factor == 0] = params["PF"] - 2 * params["PG"]

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

    return tuned_incentive_policy


def tuned_generate_individuals_for_linear(self):
    """
    Generate required params for both policies.

    Returns:
        dict: Dictionary containing parameters for both incentive and quota policies \n
        {
            "param_name": init_value_func,
        }
    """
    return {
        "PF": random.uniform(0.5, 1.5),
        "PG": random.uniform(0.1, 0.9),
        "WF_EF": random.uniform(0.05, 0.95),
        "PUR": random.uniform(0.05, 0.95),
        "SUB_BC": random.uniform(1, 12),
        "FIN_BC": random.uniform(0.1, 0.95),
        "WF_SF": random.uniform(5, 12),
        "ANT_C_F": random.uniform(0.25, 0.8),
        "ANT_N_F": random.uniform(0.05, 0.3),
        "CF": random.uniform(0.8, 2),
        "CG": random.uniform(0.15, 0.9),
    }
