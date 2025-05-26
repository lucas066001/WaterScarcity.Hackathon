import random
from scipy.ndimage import gaussian_filter1d
import numpy as np


def custom_incentive_policy(
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

    # Normalisation afin de pouvoir justifier des diffénreces de poids entre les variables
    def min_max_normalize(array, force_min=None, force_max=None):
        if force_min is not None and force_max is not None:
            min_val = force_min
            max_val = force_max
        else:
            min_val = np.min(array)
            max_val = np.max(array)

        if max_val == min_val:
            return np.zeros_like(array)
        return (array - min_val) / (max_val - min_val)

    # Initialisation du vecteur de prénalités / subventions
    fine = np.zeros(self.nb_actors)

    # On applique la normalisation
    actors_priority_norm = min_max_normalize(actors_priority)
    is_crisis_norm = min_max_normalize(is_crisis, force_min=-1, force_max=2)

    # Grâce à l'étude de correlation de la phase 1 on sait que la moyenne mobile
    # Gaussien est efficace pour prévenir des tendances futures
    def last_4_gaussian_avg(
        arr: np.ndarray, window: int = 4, sigma: float = 1.0
    ) -> float:
        if len(arr) < window:
            raise ValueError(f"Le tableau doit contenir au moins {window} éléments.")

        last_values = arr[-window:]
        weights = np.ones(window)
        gauss_weights = gaussian_filter1d(weights, sigma=sigma)
        return np.average(last_values, weights=gauss_weights)

    if len(water_flows) >= 5:
        water_flow_gauss = last_4_gaussian_avg(water_flows)
        water_flow_gauss_lag_1w = last_4_gaussian_avg(water_flows[:-1])
        # On créé un indicateur de tendance de l'eau
        # Comme on sait que cette tendance a tendance à s'inverser vis-à-vis des futurs flux d'eau
        # On pourra s'en servir pour anticiper des comportements et ajuster les pénalités
        water_flow_gauss_evol_slope = water_flow_gauss_lag_1w - water_flow_gauss
        # On divise cette valeur afin qu'elle soit dans un ordre de grandeur de 1
        # Cette valeur est arbitraire mais n'ayant pas de données comparable pour la mettre à l'échelle
        # Ce sera donc une valeur à régler en fonction de la station et des tendances hydrologiques observées
        adjusted_water_flow_gauss_evol_slope = water_flow_gauss_evol_slope / 15

    # On calcule la distance moyenne du flux d'eau par rapport au DCR
    # Cela permettra de savoir si nous sommes en période où la vigilance est nécessaire
    mean_water_flow = np.mean(water_flows)
    average_mwf_dcr_distance = mean_water_flow - DCR
    # On calcule si on est sur une distance moyenne du DCR qui est critique ou habituelle par rapport à la distance moyenne
    adjusted_mwf_dcr_distance = (water_flows[-1] - average_mwf_dcr_distance) / (
        average_mwf_dcr_distance
    )
    # Comme cette valeur peut varier entre des valeurs négatives et positives ~[-8,8]
    # On divise par 10 afin de rester dans une intervalle avec un ordre de grandeur de 1
    # Cet indicateur permettra d'anticiper les amendes et subventions avant d'atteindre les seuiles critiques
    adjusted_mwf_dcr_distance = adjusted_mwf_dcr_distance / 10

    # Une fois tous les indicateurs calculés on s'en sert pour déterminer les pénalités et subventions

    return fine


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
    params: dict = None,
) -> np.ndarray:

    # Normalisation afin de pouvoir justifier des diffénreces de poids entre les variables
    def min_max_normalize(array, force_min=None, force_max=None):
        if force_min is not None and force_max is not None:
            min_val = force_min
            max_val = force_max
        else:
            min_val = np.min(array)
            max_val = np.max(array)

        if max_val == min_val:
            return np.zeros_like(array)
        return (array - min_val) / (max_val - min_val)

    # Initialisation du vecteur de prénalités / subventions
    fine = np.zeros(self.nb_actors)

    # On applique la normalisation plus une petite valeur pour éviter les multiplications par 0
    EPSILON = 5e-2
    actors_priority_norm = min_max_normalize(actors_priority) + EPSILON
    is_crisis_norm = min_max_normalize(is_crisis, force_min=-1, force_max=2) + EPSILON
    # Niveau de crise actuel normalisé
    current_stress_level_norm = is_crisis_norm[-1]

    # Grâce à l'étude de correlation de la phase 1 on sait que la moyenne mobile
    # Gaussien est efficace pour prévenir des tendances futures
    def last_4_gaussian_avg(
        arr: np.ndarray, window: int = 4, sigma: float = 1.0
    ) -> float:
        if len(arr) < window:
            raise ValueError(f"Le tableau doit contenir au moins {window} éléments.")

        last_values = arr[-window:]
        weights = np.ones(window)
        gauss_weights = gaussian_filter1d(weights, sigma=sigma)
        return np.average(last_values, weights=gauss_weights)

    if len(water_flows) >= 5:
        water_flow_gauss = last_4_gaussian_avg(water_flows)
        water_flow_gauss_lag_1w = last_4_gaussian_avg(water_flows[:-1])
        # On créé un indicateur de tendance de l'eau
        # Comme on sait que cette tendance a tendance à s'inverser vis-à-vis des futurs flux d'eau
        # On pourra s'en servir pour anticiper des comportements et ajuster les pénalités
        water_flow_gauss_evol_slope = water_flow_gauss_lag_1w - water_flow_gauss
        # On divise cette valeur afin qu'elle soit dans un ordre de grandeur de 1
        # Cette valeur est arbitraire mais n'ayant pas de données comparable pour la mettre à l'échelle
        # Ce sera donc une valeur à régler en fonction de la station et des tendances hydrologiques observées
        adjusted_water_flow_gauss_evol_slope = water_flow_gauss_evol_slope / 15
    else:
        # Si on n'a pas assez de données pour calculer la tendance de l'eau on met 0 pour annuler son impact
        adjusted_water_flow_gauss_evol_slope = 0

    # On calcule la distance moyenne du flux d'eau par rapport au DCR
    # Cela permettra de savoir si nous sommes en période où la vigilance est nécessaire
    mean_water_flow = np.mean(water_flows)
    average_mwf_dcr_distance = mean_water_flow - DCR
    # On calcule si on est sur une distance moyenne du DCR qui est critique ou habituelle par rapport à la distance moyenne
    adjusted_mwf_dcr_distance = (water_flows[-1] - average_mwf_dcr_distance) / (
        average_mwf_dcr_distance
    )
    # Comme cette valeur peut varier entre des valeurs négatives et positives ~[-8,8]
    # On divise par 10 afin de rester dans une intervalle avec un ordre de grandeur de 1
    # Cet indicateur permettra d'anticiper les amendes et subventions avant d'atteindre les seuiles critiques
    adjusted_mwf_dcr_distance = adjusted_mwf_dcr_distance / 10

    # Cette valeur est la valeur indiquant que tout le monde pompe à son habitude et ne risque pas de passer sous le DCR
    # Il ne faut donc pas passer en dessous
    critical_overall_demand_treshold = np.sum(avg_pump) + DCR

    # Une fois tous les indicateurs calculés on s'en sert pour déterminer les pénalités et subventions
    # Objectifs :
    # - Augmenter coop en temps de crise -> Diminuer l'impact écologique
    # - Baisser la coop en temps normal -> Augmenter l'impact économique car constaté
    # qu'une trop grande coopération est néfaste en temps normal, surtout en configuration de prédictions biasées
    for i in range(self.nb_actors):

        # Définition des politiques
        CRISIS_FINE = (
            (avg_incomes[i] * params["cf_avg_incomes"])
            * (
                params["cf_actors_priority"] * (1 / actors_priority_norm[i])
            )  # Plus la priorité est grande moins la pénalité est élvée
            * (
                params["cf_curr_stress"] * (current_stress_level_norm)
            )  # Plus le stress est grand plus la pénalité est élvée
        )
        CRISIS_SUBSIDY = -(
            (avg_incomes[i] * params["cs_avg_incomes"])
            * (
                params["cs_actors_priority"] * (actors_priority_norm[i])
            )  # Plus la priorité est grande plus la subvention est élvée
            * (
                params["cs_curr_stress"] * current_stress_level_norm
            )  # Plus le stress est grand plus la subvention est élvée
        )

        # Point d'attention car si tout le monde pompe à son habitude
        # On peut passer sous le DCR
        # On commence à amorcer la politique de coopération
        # if water_flows[-1] <= critical_overall_demand_treshold:
        #     # On anticipe les mouvements des flux futurs en prenant en compte
        #     # L'inverse tendance de la moyenne mobile
        #     # Car lors de l'étude de corrélation on a vu que la tendance de la moyenne mobile
        #     # était inversement proportionnelle aux flux futurs

        #     ANTICIPATED_CRISIS_FINE = CRISIS_FINE * (
        #         params["pf_wf_slope"] * adjusted_water_flow_gauss_evol_slope
        #     )  # Plus la tendance de l'eau est à la hausse plus la pénalité est élevée

        #     ANTICIPATED_NONE_CRISIS_FINE = CRISIS_FINE * -(
        #         params["pf_wf_slope"] * adjusted_water_flow_gauss_evol_slope
        #     )  # Plus la tendance de l'eau est à la baisse plus la pénalité est faible

        #     # Non Coop
        #     if water_pump[i] > quota[i]:
        #         if adjusted_water_flow_gauss_evol_slope >= 0:
        #             # Potentiel crise à venir car hausse passée
        #             fine[i] = ANTICIPATED_CRISIS_FINE
        #         else:
        #             # Potentiel recouvrement car baisse passée
        #             fine[i] = ANTICIPATED_NONE_CRISIS_FINE
        #     # Coop
        #     else:
        #         if adjusted_water_flow_gauss_evol_slope >= 0:
        #             # Potentiel crise à venir car hausse passée
        #             fine[i] = -ANTICIPATED_CRISIS_FINE
        #         else:
        #             # Potentiel recouvrement car baisse passée
        #             fine[i] = -ANTICIPATED_NONE_CRISIS_FINE

        # Politique en temps de crise
        if current_stress_level_norm >= 1 + EPSILON:
            # Policy for actors who exceed their quota
            if water_pump[i] > quota[i]:
                fine[i] = CRISIS_FINE
            # Policy for actors who respect their quota
            elif water_pump[i] <= quota[i]:
                fine[i] = CRISIS_SUBSIDY

        # Non-crisis case
        else:
            # En temps normal on veut que les acteurs coopèrent moins
            if water_pump[i] > quota[i]:
                fine[i] = CRISIS_SUBSIDY
    if fine.max() != 0 and fine.min() != 0 and False:
        print(f"------------------- BEGIN Turn -------------------")
        print(f"max fine : {fine.max()}")
        print(f"min fine : {fine.min()}")
        print(f"current_stress_level_norm : {current_stress_level_norm}")
        print(
            f"adjusted_water_flow_gauss_evol_slope : {adjusted_water_flow_gauss_evol_slope}"
        )
        print(f"    params[cf_avg_incomes]: {params["cf_avg_incomes"]}")
        print(f"    params[cf_actors_priority]: {params["cf_actors_priority"]}")
        print(f"    params[cf_curr_stress]: {params["cf_curr_stress"]}")
        print(f"    params[cs_avg_incomes]: {params["cs_avg_incomes"]}")
        print(f"    params[cs_actors_priority]: {params["cs_actors_priority"]}")
        print(f"    params[cs_curr_stress]: {params["cs_curr_stress"]}")
        print(f"    params[pf_wf_slope]: {params["pf_wf_slope"]}")

        print(f"------------------- END Turn -------------------")
        import time

        time.sleep(0.5)
    return fine


def generate_individuals(self):
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
            # crisis params
            "defect_income_factor": random.uniform(0.01, 1),
            "defect_prio_factor": random.uniform(0.01, 2),
            "coop_fine_inc": random.uniform(0.1, 2),
            "coop_fine_prio": random.uniform(0.9, 3),
            "coop_fine_stress": random.uniform(0.1, 5),
            "coop_sub_inc_factor": random.uniform(0.1, 2),
            "coop_sub_prio": random.uniform(0.1, 3),
            "coop_sub_stress": random.uniform(0.9, 5),
        },
        "quota_params": {
            "avg_pump_factor": random.uniform(0.01, 2),
            "actor_priority_factor": random.uniform(0.01, 2),
            "crisis_level_factor": random.uniform(0.01, 3),
        },
    }


def make_quota_function(self, params):
    """
    Based on given parameters it generate a quota policy.

    Returns:
        func: Quota policy.
    """

    def custom_quota(self, crisis_level, actors_priority, avg_pump, DOE, DCR):
        def min_max_normalize(array, force_min=None, force_max=None):
            if force_min is not None and force_max is not None:
                min_val = force_min
                max_val = force_max
            else:
                min_val = np.min(array)
                max_val = np.max(array)

            if max_val == min_val:
                return np.zeros_like(array)
            return (array - min_val) / (max_val - min_val)

        EPISLON = 5e-2
        actors_priority_norm = min_max_normalize(actors_priority) + EPISLON
        crisis_level_norm = (
            min_max_normalize(np.array([crisis_level]), force_min=-1, force_max=2)
            + EPISLON
        )

        quota = (
            params["avg_pump_factor"]
            * avg_pump
            * params["actor_priority_factor"]
            * actors_priority_norm
            * params["crisis_level_factor"]
            * crisis_level_norm[0]
        )
        # Forcer des quotas réalistes (>= 0)
        return np.clip(quota, 0, avg_pump)

    return custom_quota


def make_incentive_function(self, params):
    """
    Based on given parameters it generate an incentive policy.

    Returns:
        func: Incentive policy.
    """

    def custom_incentive(
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
    ):
        # Normalisation afin de pouvoir justifier des diffénreces de poids entre les variables
        def min_max_normalize(array, force_min=None, force_max=None):
            if force_min is not None and force_max is not None:
                min_val = force_min
                max_val = force_max
            else:
                min_val = np.min(array)
                max_val = np.max(array)

            if max_val == min_val:
                return np.zeros_like(array)
            return (array - min_val) / (max_val - min_val)

        # Initialisation du vecteur de prénalités / subventions
        fine = np.zeros(self.nb_actors)

        # On applique la normalisation plus une petite valeur pour éviter les multiplications par 0
        EPSILON = 5e-2
        actors_priority_norm = min_max_normalize(actors_priority) + EPSILON
        is_crisis_norm = (
            min_max_normalize(is_crisis, force_min=-1, force_max=2) + EPSILON
        )
        # Niveau de crise actuel normalisé
        current_stress_level_norm = is_crisis_norm[-1]

        # Cette valeur est la valeur indiquant que tout le monde pompe à son habitude et ne risque pas de passer sous le DCR
        # Il ne faut donc pas passer en dessous
        critical_overall_demand_treshold = np.sum(avg_pump) + DCR

        # Une fois tous les indicateurs calculés on s'en sert pour déterminer les pénalités et subventions
        # Objectifs :
        # - Augmenter coop en temps de crise -> Diminuer l'impact écologique
        # - Baisser la coop en temps normal -> Augmenter l'impact économique car constaté
        # qu'une trop grande coopération est néfaste en temps normal, surtout en configuration de prédictions biasées
        for i in range(self.nb_actors):

            # Politique en temps normal
            if current_stress_level_norm < 1 + EPSILON:
                if water_pump[i] > quota[i]:
                    # On encourage la NON coopération en temps normal via des subventions uniquement
                    fine[i] = -(
                        ((avg_incomes[i] / 5) * params["defect_income_factor"])
                        * (
                            # Subvention proportionelle au 5ème des revenu pour moins d'impact écolo
                            params["defect_prio_factor"]
                            * (actors_priority_norm[i])
                        )  # Plus la priorité est grande moins la subvention est élvée
                    )
            # Politique en temps de crise
            else:
                # On encourage la coopération en temps de crise
                if water_pump[i] > quota[i]:
                    fine[i] = (
                        ((avg_incomes[i] / 10) * params["coop_fine_inc"])
                        * (
                            # Amende proportionelle au 10ème des revenu pour moins d'impact éco
                            params["coop_fine_prio"]
                            * (1 / actors_priority_norm[i])
                        )
                        * (  # Plus la priorité est grande moins la pénalité est élvée
                            params["coop_fine_stress"] * current_stress_level_norm
                        )
                    )  # Plus le stress est grand plus la pénalité est élvée
                else:
                    fine[i] = -(
                        (avg_incomes[i] * params["coop_sub_inc_factor"])
                        * (
                            # Subvention proportionelle au revenu pour plus d'impact
                            params["coop_sub_prio"]
                            * (actors_priority_norm[i])
                        )
                        * (  # Plus la priorité est grande plus la subvention est élvée
                            params["coop_sub_stress"] * current_stress_level_norm
                        )  # Plus le stress est grand plus la subvention est élvée
                    )

        if fine.max() != 0 and fine.min() != 0 and False:
            print(f"------------------- BEGIN Turn -------------------")
            print(f"max fine : {fine.max()}")
            print(f"min fine : {fine.min()}")
            print(f"current_stress_level_norm : {current_stress_level_norm}")
            print(
                f"adjusted_water_flow_gauss_evol_slope : {adjusted_water_flow_gauss_evol_slope}"
            )
            print(f"    params[cf_avg_incomes]: {params["cf_avg_incomes"]}")
            print(f"    params[cf_actors_priority]: {params["cf_actors_priority"]}")
            print(f"    params[cf_curr_stress]: {params["cf_curr_stress"]}")
            print(f"    params[cs_avg_incomes]: {params["cs_avg_incomes"]}")
            print(f"    params[cs_actors_priority]: {params["cs_actors_priority"]}")
            print(f"    params[cs_curr_stress]: {params["cs_curr_stress"]}")
            print(f"    params[pf_wf_slope]: {params["pf_wf_slope"]}")

            print(f"------------------- END Turn -------------------")
            import time

            time.sleep(0.5)
        return fine

    return custom_incentive
