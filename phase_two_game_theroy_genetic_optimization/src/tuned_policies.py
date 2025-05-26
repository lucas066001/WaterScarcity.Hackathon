# Define quota policy - determines how much water each actor is allowed to pump
# This uses the hard_quota policy from the quota_policies module
# In crisis situations, actors with priority below the crisis level get zero quota
class CrisisLevel:
    NORMAL = -1
    ALERT = 0
    CRISIS = 1
    EXTREME_CRISIS = 2


def tuned_make_quota_function(self, params):
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
        proportional_priority_factor = 1 + ((actors_priority + 1) / 10)
        # print("crisis_level", crisis_level)
        if crisis_level == CrisisLevel.NORMAL:
            # Actors will be assigned bigger quotas to avoid them getting penalties and so pumping more
            return avg_pump * params["NORMAL_GROWTH_FACTOR"]

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

            estimated_flow = maximum_amount * params["ESTIMATION_FACTOR"]

            # Repartition by actor priority
            # Sort actors by priority, get indices
            sorted_indices = np.argsort(-actors_priority)
            quotas = np.zeros_like(avg_pump)

            for idx in sorted_indices:
                if estimated_flow <= DCR:
                    # If the remaining water is less than DCR, stop allocating, default value already 0
                    break
                pump = (
                    avg_pump[idx]
                    * params["CRISIS_PUMPING_RESTRICTION"]
                    * proportional_priority_factor[idx]
                )
                if estimated_flow - pump >= DCR:
                    quotas[idx] = pump
                    estimated_flow -= pump
                else:
                    # If the remaining water is less than DCR the actor can't pump
                    quotas[idx] = 0

            return quotas

    return tuned_quota


def tuned_make_incentive_function(self, params):
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

        # Defining a custom treshold that represent an anticipation crisis point
        critical_overall_demand_treshold = np.sum(avg_pump) + DCR

        # Adjusting so the priority value is [0,1] range
        actor_priority = (actors_priority - 0) / (2 - 0)
        # Epsilon [0,0.5] param will help to select the right proportion of policy increase in [0,1] range so the inverse function will be stricly positive
        # Higher the priority is, lower the factor will be, but factor will still be > 1
        adjusted_priority = actor_priority + params["EPSILON_PRIORITY"]
        adjusted_priority_factor = (1 / adjusted_priority) / 10

        # As the simulation penalizes less subsidies we push more on it than fines
        SUBSIDY = -(
            # We make 5 times bigger subsidy as the simulation penalize 5 times less
            # So our politics align with the game rules and prefer pushing coop through subsidies rather than fines
            (DCR * params["SUBSIDY_DCR_BASE_CALCULATION"])
            * adjusted_priority_factor
        )
        FINE = (
            avg_incomes * params["FINE_INCOME_BASE_CALCULATION"]
        ) * adjusted_priority_factor

        # print("------------------")
        # print("actors prio:", actors_priority)
        # print("adjusted_priority_factor:", adjusted_priority_factor)
        # print("avg_incomes:", avg_incomes)
        # print("params[FINE_INCOME_BASE_CALCULATION]:", params["FINE_INCOME_BASE_CALCULATION"])
        # print("FINE", FINE)
        # print("SUBSIDY", SUBSIDY)
        # print("------------------")

        exceding_quota_idx = water_pump > quota
        respecting_quota_idx = water_pump <= quota

        if (
            water_flows[-1] < critical_overall_demand_treshold
            and crisis_level == CrisisLevel.NORMAL
        ):
            # We anticipate a near crisis situation so we start applying fines
            # But it won't be as strong as in a real crisis

            fine[exceding_quota_idx] = (
                FINE[exceding_quota_idx] * params["ANTICIPATION_FACTOR"]
            )

            fine[respecting_quota_idx] = (
                SUBSIDY[respecting_quota_idx] * params["ANTICIPATION_FACTOR"]
            )
        else:

            match crisis_level:
                case CrisisLevel.NORMAL:
                    # For the moment, normal times will be free from fines and subsidies
                    return np.zeros_like(avg_pump)

                case (
                    CrisisLevel.ALERT | CrisisLevel.CRISIS | CrisisLevel.EXTREME_CRISIS
                ):
                    # The more the crisis is critical, the more the fine/subsidies will be
                    # [1.2, 1.3, 1.4]
                    crisis_factor = 1 + ((crisis_level + 2) / 10)

                    actors_priority_below_crisis = actors_priority < crisis_level
                    actors_priority_above_crisis = actors_priority >= crisis_level

                    # Defectors
                    # Below priority and exceeding
                    actors_exceding_and_below_priority_idx = (
                        actors_priority_below_crisis == exceding_quota_idx
                    )
                    fine[actors_exceding_and_below_priority_idx] = (
                        FINE[actors_exceding_and_below_priority_idx]
                        * params["NON_PRIORITY_ACTORS_INCREASE"]
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
                        * params["NON_PRIORITY_ACTORS_INCREASE"]
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

    return tuned_incentive_policy


def tuned_generate_individuals(self):
    """
    Generate required params for both policies.

    Returns:
        dict: Dictionary containing parameters for both incentive and quota policies \n
        {
            "incentive_params":{...},\n
            "quota_params": {...}
        }
    """
    return {
        "incentive_params": {
            # The lower it is, the less anticipation fines and subsidies will be
            #   ]0, 2]
            #   e.g : 0.6 => 60% of the real crisis fine
            "ANTICIPATION_FACTOR": random.uniform(0.35, 1.6),
            # The higher it is, more the low priority actors breaking/respecting
            # Quota and priority rule will be penalized/rewarded, this aims to keep priority order
            #   [1, 2]
            #   e.g : 1.2 => 120% of initial fine/subsidy
            "NON_PRIORITY_ACTORS_INCREASE": random.uniform(1, 2),
            # This will indicate how strong will be the additional subsidy/fine
            # When low priority actors exceed their quota in crisis times
            #   ]0, 1]
            #   e.g : 0.10 => 10% of the average income
            "SUBSIDY_DCR_BASE_CALCULATION": random.uniform(0.05, 2),
            # This will indicate on wich amount the fine will be calculated
            # The higher it is, the closer it gets to the actor average income
            #   ]0, 1]
            #   e.g : 0.10 => 10% of the average income
            "FINE_INCOME_BASE_CALCULATION": random.uniform(0.001, 0.8),
            # This will accentuate the exponentinality of subsidy/fine policy based on actor priority
            #   ]0, 1]
            #   e.g : 0.25 => Closer to zero, applied to inverse function, will impact P0(0.25 -> 1/0.25=4) 5 times more than P1(1.25 -> 1/1.25=0.8)
            #   e.g : 0.9 => Closer to one, applied to inverse function, will impact P0(0.9 -> 1/0.9=1.1) ~2.1 times more than P1(1.9 -> 1/1.9=0.52)
            "EPSILON_PRIORITY": random.uniform(0.01, 0.8),
        },
        "quota_params": {
            # The bigger it is, more the actors will be allowed to pump in normal times
            #   [1, +inf]
            #   e.g : 1.15 => 15% more than the average pump
            "NORMAL_GROWTH_FACTOR": random.uniform(0.90, 1.5),
            # The lower it is, the more pessimistic the allocation will be
            #   ]0, 1]
            #   1 => maximum estimation possible (e.g : CrisisLevel.ALERT = DOE )
            #   0.05 => maximum estimation * 0.05 (e.g : CrisisLevel.ALERT = 5% of DOE )
            "ESTIMATION_FACTOR": random.uniform(0.05, 1),
            # The lower it is, the less actors will be allowed to pump in crisis times
            #   [0, 1]
            #   1 => 100% of average pump
            #   0 => 0% of average pump
            "CRISIS_PUMPING_RESTRICTION": random.uniform(0.05, 1),
        },
    }
